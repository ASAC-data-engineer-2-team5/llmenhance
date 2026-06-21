# AWS Deployment

This deployment uses AWS for the application runtime while keeping Ollama/Qwen as an on-premise-like model server.

## Contract

- Qwen is used only for RAG answer generation.
- Qwen answers only from retrieved internal document chunks.
- If retrieved context does not confirm the answer, the service must answer `문서에서 확인되지 않습니다`.
- Every grounded answer must include source references.
- Ollama/Qwen is not installed inside Docker for the MVP.
- The existing Ollama/Qwen EC2 is treated as the model server.
- Terraform manages the app-side AWS resources first.
- Port `11434` must not be opened to the internet.
- Qdrant must not be exposed publicly.
- The MVP app endpoint is verified through SSM port forwarding first.
- Temporary public Streamlit exposure is allowed for demos by opening only TCP `8501`.

## Runtime Shape

```text
Tester workstation
-> public Streamlit URL or AWS SSM port forward
-> app EC2 Streamlit:8501 or localhost:8000
-> Docker Compose streamlit/rag-api/qdrant
-> existing EC2 Ollama/Qwen model server
```

## First Deploy

```bash
cd /opt/llmenhance/app
docker compose -f docker-compose.aws.yml up -d --build
docker compose -f docker-compose.aws.yml run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
./scripts/aws_verify.sh
```

## Product Runtime

The AWS MVP deployment uses the Qwen-only Streamlit surface by default:

```env
ENABLE_GEMINI_PANEL=false
ENABLE_GEMINI_ENDPOINT=false
```

The Gemini comparison panel is allowed only for internal benchmark sessions.

## CI/CD

This repository includes two GitHub Actions workflows for the AWS MVP deployment.

### Terraform CI

`.github/workflows/terraform-ci.yml` runs on pull requests to `main` and pushes to `main` when AWS deployment files change.

It always validates:

- Terraform formatting under `infra/terraform`.
- Terraform syntax for `infra/terraform/bootstrap`.
- Terraform syntax for `infra/terraform/envs/mvp`.
- `docker-compose.aws.yml` rendering.

It also runs a remote-state Terraform plan when these repository variables are configured:

```text
AWS_REGION=ap-northeast-2
AWS_ROLE_TO_ASSUME=arn:aws:iam::<account-id>:role/<github-actions-role>
TF_STATE_BUCKET=<terraform-state-bucket>
OLLAMA_BASE_URL=http://<model-server-ip>:11434
STREAMLIT_ALLOWED_CIDR_BLOCKS=["0.0.0.0/0"]
```

`STREAMLIT_ALLOWED_CIDR_BLOCKS` should be restricted for non-demo use.

### Manual MVP CD

`.github/workflows/deploy-mvp.yml` is a manual `workflow_dispatch` deployment workflow. GitHub shows this manual workflow after the workflow file exists on the default branch.

The workflow:

1. Assumes the AWS role through GitHub OIDC.
2. Initializes Terraform with the S3 backend.
3. Optionally applies Terraform.
4. Reads the app EC2 instance id from Terraform output.
5. Uses SSM Run Command to deploy the requested Git ref on the app EC2.
6. Optionally reindexes `datasets/docs`.
7. Optionally runs `scripts/aws_verify.sh` on the app EC2.

Recommended manual inputs:

```text
git_ref=main
apply_terraform=false
reindex=true
run_verify=true
```

Set `apply_terraform=true` only when infrastructure changes should be applied from the workflow.

### GitHub OIDC Role

The deployment workflows expect an AWS IAM role that trusts GitHub Actions OIDC and can be assumed by this repository. Prefer GitHub OIDC over long-lived AWS access keys.

At minimum, the role needs permissions for:

- S3 backend state read/write for Terraform.
- AWS resources managed by `infra/terraform/envs/mvp` when `apply_terraform=true`.
- SSM `SendCommand`, `GetCommandInvocation`, and related read actions for the app EC2.

Use a GitHub Environment named `mvp` with required reviewers before allowing production-like deployments.

## Access

When `streamlit_allowed_cidr_blocks` includes the viewer's IP range, open the public frontend:

```text
http://<app-public-ip>:8501
```

Use Session Manager port forwarding for private MVP validation or API checks:

```bash
aws ssm start-session \
  --target <app-instance-id> \
  --document-name AWS-StartPortForwardingSession \
  --parameters "portNumber=8501,localPortNumber=8501"
```

Open `http://localhost:8501` on the operator workstation.

## Backup

- Snapshot the app EC2 root EBS volume before any destructive re-indexing.
- Use AWS Data Lifecycle Manager for daily EBS snapshots during MVP testing.
- Keep at least seven daily snapshots.
- Do not commit Qdrant data, SQLite files, `.env`, or Terraform state files.

## Logging

- CloudWatch Agent may collect host metrics, Docker logs, disk usage, and memory usage.
- Application logs must not include full confidential document chunks by default.
- The sample RAG answer can be logged only with answer metadata and source ids.

## Rollback

```bash
cd /opt/llmenhance/app
git fetch --all --prune
git checkout <last-good-commit-or-branch>
docker compose -f docker-compose.aws.yml up -d --build
./scripts/aws_verify.sh
```

## Public Exposure Gates

Do not expose `/api/ask/qwen`, Qdrant, or Ollama to the public internet. For Streamlit, use public exposure only as a temporary MVP demo path until all are complete:

- Authentication exists.
- HTTPS exists.
- CORS is restricted to the real frontend origin.
- Request body size and rate limits exist.
- Port `11434` remains private.
- Port `6333` remains private.
- Product UI is Qwen-only by default.
