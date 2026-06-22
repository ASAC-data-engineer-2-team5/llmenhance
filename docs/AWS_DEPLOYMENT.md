# AWS Existing EC2 Deployment

This project deploys the current `main` branch to the existing Osaka EC2 instance
`i-0ccf9071972894f30`.

The CD workflow does not create EC2 infrastructure. It uses GitHub Actions OIDC
to call AWS SSM Run Command against the existing instance.

## One-Time EC2 Requirements

- Instance region: `ap-northeast-3`
- Instance id: `i-0ccf9071972894f30`
- SSM agent is installed and online.
- The instance role allows `AmazonSSMManagedInstanceCore`.
- Docker and Docker Compose v2 are installed.
- Host Ollama is running on port `11434`.
- Existing EC2 uses `OLLAMA_EMBEDDING_TIMEOUT_SECONDS=180` because cold
  `bge-m3` embedding calls can exceed the local 30 second default.
- Gemini credential JSON exists at:

```bash
/home/ubuntu/secrets/google-application-credentials.json
```

## GitHub Environment Variables

Configure these variables in the `mvp` environment:

```text
AWS_REGION=ap-northeast-3
AWS_APP_INSTANCE_ID=i-0ccf9071972894f30
AWS_ROLE_TO_ASSUME=the exact IAM role ARN trusted by GitHub Actions OIDC
GOOGLE_CLOUD_PROJECT=the exact GCP project id with Vertex Gemini enabled
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-2.5-flash
GEMINI_THINKING_BUDGET=0
BEDROCK_REGION=ap-northeast-2
BEDROCK_MODEL_ID=
```

## Deploy

Run GitHub Actions workflow `Deploy Existing EC2`.

Recommended inputs:

```text
git_ref=main
reindex=true
run_verify=true
```

The deployed app lives at:

```bash
/opt/llmenhance/app
```

The workflow avoids the user worktree at:

```bash
/home/ubuntu/eunbee/llmenhance
```

## Verify On EC2

```bash
cd /opt/llmenhance/app
./scripts/aws_verify.sh
```

Expected final line:

```text
AWS_VERIFY_OK
```
