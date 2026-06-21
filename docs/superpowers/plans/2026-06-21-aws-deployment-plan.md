# AWS Terraform Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the current `origin/main` llmenhance MVP on AWS with Terraform while treating the already-installed EC2 Ollama/Qwen server as the on-premise model server.

**Architecture:** Terraform manages the application-side AWS resources first: one app EC2 instance, an Elastic IP for model-server allowlisting, IAM/SSM access, security groups, and AWS operational tags. The existing Ollama/Qwen EC2 is not recreated in the first release; the app stack reaches it through `OLLAMA_BASE_URL`, and Docker runs only `rag-api`, `streamlit`, and `qdrant`. Public exposure is blocked until authentication and HTTPS are added.

**Tech Stack:** Terraform `>= 1.10`, AWS provider `~> 6.0`, S3 backend with native lockfile, EC2, IAM, SSM Session Manager, Docker Compose, FastAPI, Streamlit, Qdrant, EBS, existing Ollama/Qwen model server.

---

## Basis

This plan is based on `origin/main` commit `05e4597ad5bca41c030ad6645ee1a6e19f16d443`.

Current main facts:

- `docker-compose.yml` runs `qdrant`, `rag-api`, and `streamlit`.
- `rag-api` runs `uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload`.
- Streamlit runs `frontend/streamlit_app.py` on port `8501`.
- Qwen API route is `POST /api/ask/qwen`.
- Gemini comparison route is `POST /api/ask/gemini`; this is not part of the Qwen-only product runtime.
- `app.qwen_client.chat_qwen` keeps Ollama chat messages separated as system/user messages and adds the prompt injection guard.
- `app.rag_pipeline` returns `"문서에서 확인되지 않습니다"` when no retrieved context can ground the answer.
- `.env.shared-ec2.example` currently points at the shared EC2 Ollama endpoint.

External references checked:

- Terraform `aws_instance`: https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/instance
- Terraform S3 backend locking: https://developer.hashicorp.com/terraform/language/backend/s3
- AWS Session Manager: https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html
- AWS security groups: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-security-groups.html
- AWS CloudWatch Agent: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Install-CloudWatch-Agent.html
- AWS Data Lifecycle Manager for EBS snapshots: https://docs.aws.amazon.com/dlm/

## Deployment Contract

```text
Existing EC2 with Ollama/Qwen
  = model server / on-premise assumption
  = not recreated by Terraform in the first release
  = no public 11434 access except approved app/tester networks

Terraform-created app EC2
  = Docker Compose host for qdrant, rag-api, streamlit
  = calls existing model server through OLLAMA_BASE_URL
  = reached by SSM port forwarding for MVP validation

Employee-facing public endpoint
  = blocked until auth, HTTPS, request limits, and non-wildcard CORS exist
```

## File Structure

Create or modify these files:

- Create `docs/AWS_DEPLOYMENT.md`: operator runbook and deployment contract.
- Create `.env.aws-ec2.example`: AWS app runtime environment.
- Create `docker-compose.aws.yml`: production-like Compose file without source bind mounts or reload.
- Create `scripts/aws_verify.sh`: EC2-local verification script.
- Create `infra/terraform/bootstrap/versions.tf`: Terraform and provider constraints for state bootstrap.
- Create `infra/terraform/bootstrap/main.tf`: S3 backend bucket bootstrap.
- Create `infra/terraform/bootstrap/outputs.tf`: backend output values.
- Create `infra/terraform/envs/mvp/versions.tf`: Terraform and AWS provider constraints.
- Create `infra/terraform/envs/mvp/backend.tf`: backend block using S3 lockfile.
- Create `infra/terraform/envs/mvp/variables.tf`: environment inputs.
- Create `infra/terraform/envs/mvp/main.tf`: EC2, Elastic IP, IAM, and security group resources.
- Create `infra/terraform/envs/mvp/user_data_app.sh.tftpl`: app host bootstrap script.
- Create `infra/terraform/envs/mvp/outputs.tf`: connection and SSM tunnel outputs.
- Create `infra/terraform/envs/mvp/terraform.tfvars.example`: safe default MVP values.
- Modify `.gitignore`: ignore local Terraform state and local tfvars.
- Optional modify `frontend/streamlit_app.py`: hide Gemini panel for product deployment.

---

### Task 1: Add The AWS Runtime Contract

**Files:**
- Create: `docs/AWS_DEPLOYMENT.md`
- Create: `.env.aws-ec2.example`
- Create: `docker-compose.aws.yml`
- Create: `scripts/aws_verify.sh`
- Modify: `.gitignore`

- [ ] **Step 1: Create `docs/AWS_DEPLOYMENT.md`**

````markdown
# AWS Deployment

This deployment uses AWS for the application runtime while keeping Ollama/Qwen as an on-premise-like model server.

## Contract

- Qwen is used only for RAG answer generation.
- Qwen answers only from retrieved internal document chunks.
- If retrieved context does not confirm the answer, the service must answer `문서에서 확인되지 않습니다`.
- Every answer must include source references.
- Ollama/Qwen is not installed inside Docker for the MVP.
- The existing Ollama/Qwen EC2 is treated as the model server.
- Terraform manages the app-side AWS resources first.
- Port `11434` must not be opened to the internet.
- Qdrant must not be exposed publicly.
- The MVP app endpoint is verified through SSM port forwarding first.

## Runtime Shape

```text
Tester workstation
-> AWS SSM port forward
-> app EC2 localhost:8501 or localhost:8000
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
````

- [ ] **Step 2: Create `.env.aws-ec2.example`**

```env
TEAM_ENV_PROFILE=aws-app-ec2
OLLAMA_BASE_URL=
LLM_MODEL=qwen3:4b-instruct
EMBEDDING_MODEL=bge-m3

QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=llmenhance_chunks

RETRIEVAL_TOP_K=5
TEMPERATURE=0.2
NUM_CTX=4096
NUM_PREDICT=512

ENABLE_GEMINI_ENDPOINT=false
ENABLE_GEMINI_PANEL=false
```

- [ ] **Step 3: Create `docker-compose.aws.yml`**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "127.0.0.1:6333:6333"
    volumes:
      - /opt/llmenhance/qdrant:/qdrant/storage
    restart: unless-stopped

  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - qdrant
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - /opt/llmenhance/storage:/app/storage
    working_dir: /app
    command: uvicorn app.server:app --host 0.0.0.0 --port 8000
    restart: unless-stopped

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile
    env_file:
      - .env
    environment:
      RAG_API_URL: http://rag-api:8000
    depends_on:
      - rag-api
    ports:
      - "127.0.0.1:8501:8501"
    working_dir: /app
    command: streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
    restart: unless-stopped
```

- [ ] **Step 4: Create `scripts/aws_verify.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

curl -fsS http://127.0.0.1:6333 >/dev/null
curl -fsS http://127.0.0.1:8000/health >/dev/null
curl -fsS http://127.0.0.1:8501/_stcore/health >/dev/null

docker compose -f docker-compose.aws.yml run --rm rag-api python -m app.healthcheck

curl -fsS http://127.0.0.1:8000/api/ask/qwen \
  -H 'content-type: application/json' \
  -d '{"question":"연차 신청은 며칠 전까지 해야 하나요?","top_k":3}' \
  -o /tmp/llmenhance-qwen-response.json

python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/tmp/llmenhance-qwen-response.json").read_text(encoding="utf-8"))
if not payload.get("sources"):
    raise SystemExit("expected non-empty sources for grounded sample answer")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
```

- [ ] **Step 5: Modify `.gitignore`**

Add:

```gitignore
# Terraform local state and local values
infra/terraform/**/.terraform/
infra/terraform/**/*.tfstate
infra/terraform/**/*.tfstate.*
infra/terraform/**/terraform.tfvars
infra/terraform/**/backend.hcl
```

- [ ] **Step 6: Run focused verification**

```powershell
docker compose -f docker-compose.yml config
docker compose -f docker-compose.aws.yml config
git diff --check
```

Expected:

```text
Both compose files render successfully.
git diff --check exits with code 0.
```

- [ ] **Step 7: Commit**

```bash
git add docs/AWS_DEPLOYMENT.md .env.aws-ec2.example docker-compose.aws.yml scripts/aws_verify.sh .gitignore
git commit -m "chore: add AWS EC2 runtime contract"
```

---

### Task 2: Add Terraform Remote State Bootstrap

**Files:**
- Create: `infra/terraform/bootstrap/versions.tf`
- Create: `infra/terraform/bootstrap/main.tf`
- Create: `infra/terraform/bootstrap/outputs.tf`

- [ ] **Step 1: Create `infra/terraform/bootstrap/versions.tf`**

```hcl
terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.7"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
```

- [ ] **Step 2: Create `infra/terraform/bootstrap/main.tf`**

```hcl
variable "aws_region" {
  type    = string
  default = "ap-northeast-2"
}

variable "project_name" {
  type    = string
  default = "llmenhance"
}

variable "environment" {
  type    = string
  default = "mvp"
}

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "terraform_state" {
  bucket = "${var.project_name}-${var.environment}-tfstate-${random_id.suffix.hex}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket                  = aws_s3_bucket.terraform_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

- [ ] **Step 3: Create `infra/terraform/bootstrap/outputs.tf`**

```hcl
output "state_bucket" {
  value = aws_s3_bucket.terraform_state.bucket
}

output "state_region" {
  value = var.aws_region
}
```

- [ ] **Step 4: Run bootstrap**

```powershell
cd infra/terraform/bootstrap
terraform init
terraform fmt -recursive
terraform validate
terraform apply
```

Expected:

```text
Apply complete!
state_bucket prints a bucket name starting with llmenhance-mvp-tfstate-
```

- [ ] **Step 5: Commit**

```bash
git add infra/terraform/bootstrap
git commit -m "infra: add terraform state bootstrap"
```

---

### Task 3: Add Terraform MVP App Stack

**Files:**
- Create: `infra/terraform/envs/mvp/versions.tf`
- Create: `infra/terraform/envs/mvp/backend.tf`
- Create: `infra/terraform/envs/mvp/variables.tf`
- Create: `infra/terraform/envs/mvp/main.tf`
- Create: `infra/terraform/envs/mvp/outputs.tf`
- Create: `infra/terraform/envs/mvp/terraform.tfvars.example`

- [ ] **Step 1: Create `infra/terraform/envs/mvp/versions.tf`**

```hcl
terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
```

- [ ] **Step 2: Create `infra/terraform/envs/mvp/backend.tf`**

```hcl
terraform {
  backend "s3" {
    key          = "llmenhance/mvp/terraform.tfstate"
    region       = "ap-northeast-2"
    use_lockfile = true
  }
}
```

Initialize with the bootstrap bucket:

```powershell
cd infra/terraform/bootstrap
$bucket = terraform output -raw state_bucket
cd ../envs/mvp
@"
bucket = "$bucket"
"@ | Set-Content -Encoding utf8 backend.hcl
terraform init "-backend-config=backend.hcl"
```

- [ ] **Step 3: Create `infra/terraform/envs/mvp/variables.tf`**

```hcl
variable "aws_region" {
  type    = string
  default = "ap-northeast-2"
}

variable "project_name" {
  type    = string
  default = "llmenhance"
}

variable "environment" {
  type    = string
  default = "mvp"
}

variable "vpc_id" {
  type    = string
  default = ""
}

variable "subnet_id" {
  type    = string
  default = ""
}

variable "ami_id" {
  type    = string
  default = ""
}

variable "instance_type" {
  type    = string
  default = "t3.large"
}

variable "root_volume_gb" {
  type    = number
  default = 100
}

variable "repo_url" {
  type    = string
  default = "https://github.com/ASAC-data-engineer-2-team5/llmenhance.git"
}

variable "repo_ref" {
  type    = string
  default = "main"
}

variable "ollama_base_url" {
  type = string

  validation {
    condition     = length(trimspace(var.ollama_base_url)) > 0 && (startswith(var.ollama_base_url, "http://") || startswith(var.ollama_base_url, "https://"))
    error_message = "ollama_base_url must be a non-empty http:// or https:// URL for the existing model server."
  }
}
```

- [ ] **Step 4: Create `infra/terraform/envs/mvp/main.tf`**

```hcl
data "aws_vpc" "default" {
  count   = var.vpc_id == "" ? 1 : 0
  default = true
}

locals {
  vpc_id = var.vpc_id == "" ? data.aws_vpc.default[0].id : var.vpc_id
}

data "aws_subnets" "default" {
  count = var.subnet_id == "" ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [local.vpc_id]
  }
}

data "aws_ami" "ubuntu" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

locals {
  subnet_id = var.subnet_id == "" ? tolist(data.aws_subnets.default[0].ids)[0] : var.subnet_id
  ami_id    = var.ami_id == "" ? data.aws_ami.ubuntu[0].id : var.ami_id

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_iam_role" "app" {
  name = "${var.project_name}-${var.environment}-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ec2.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "cloudwatch" {
  role       = aws_iam_role.app.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

resource "aws_iam_instance_profile" "app" {
  name = "${var.project_name}-${var.environment}-app-profile"
  role = aws_iam_role.app.name
}

resource "aws_security_group" "app" {
  name_prefix = "${var.project_name}-${var.environment}-app-"
  description = "llmenhance app EC2 security group"
  vpc_id      = local.vpc_id

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.environment}-app"
  })
}

resource "aws_vpc_security_group_egress_rule" "all" {
  security_group_id = aws_security_group.app.id
  ip_protocol       = "-1"
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_instance" "app" {
  ami                         = local.ami_id
  instance_type               = var.instance_type
  subnet_id                   = local.subnet_id
  vpc_security_group_ids      = [aws_security_group.app.id]
  iam_instance_profile        = aws_iam_instance_profile.app.name
  associate_public_ip_address = true

  user_data = templatefile("${path.module}/user_data_app.sh.tftpl", {
    repo_url        = var.repo_url
    repo_ref        = var.repo_ref
    ollama_base_url = var.ollama_base_url
  })
  user_data_replace_on_change = true

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true

    tags = merge(local.tags, {
      Name   = "${var.project_name}-${var.environment}-app-root"
      Backup = "${var.project_name}-${var.environment}"
    })
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.environment}-app"
  })
}

resource "aws_eip" "app" {
  domain   = "vpc"
  instance = aws_instance.app.id

  tags = merge(local.tags, {
    Name = "${var.project_name}-${var.environment}-app-eip"
  })
}
```

- [ ] **Step 5: Create `infra/terraform/envs/mvp/outputs.tf`**

```hcl
output "app_instance_id" {
  value = aws_instance.app.id
}

output "app_public_ip" {
  value = aws_eip.app.public_ip
}

output "ssm_streamlit_tunnel_command" {
  value = "aws ssm start-session --target ${aws_instance.app.id} --document-name AWS-StartPortForwardingSession --parameters '{\"portNumber\":[\"8501\"],\"localPortNumber\":[\"8501\"]}'"
}

output "ssm_api_tunnel_command" {
  value = "aws ssm start-session --target ${aws_instance.app.id} --document-name AWS-StartPortForwardingSession --parameters '{\"portNumber\":[\"8000\"],\"localPortNumber\":[\"8000\"]}'"
}
```

- [ ] **Step 6: Create `infra/terraform/envs/mvp/terraform.tfvars.example`**

```hcl
aws_region              = "ap-northeast-2"
project_name            = "llmenhance"
environment             = "mvp"
instance_type           = "t3.large"
root_volume_gb          = 100
repo_url                = "https://github.com/ASAC-data-engineer-2-team5/llmenhance.git"
repo_ref                = "main"
```

Create local `terraform.tfvars` from this example and add the approved model server URL:

```hcl
ollama_base_url = "http://10.0.0.25:11434"
```

- [ ] **Step 7: Validate Terraform**

```powershell
cd infra/terraform/envs/mvp
Copy-Item terraform.tfvars.example terraform.tfvars
terraform fmt -recursive
terraform validate
terraform plan
```

Expected:

```text
terraform validate exits successfully.
terraform plan proposes one EC2 instance, one Elastic IP, one security group, one IAM role, one instance profile, and related attachments.
No inbound app port is opened.
```

- [ ] **Step 8: Commit**

```bash
git add infra/terraform/envs/mvp
git commit -m "infra: add MVP app EC2 terraform stack"
```

---

### Task 4: Add EC2 App Bootstrap

**Files:**
- Create: `infra/terraform/envs/mvp/user_data_app.sh.tftpl`

- [ ] **Step 1: Create `infra/terraform/envs/mvp/user_data_app.sh.tftpl`**

```bash
#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl git docker.io docker-compose-v2 python3
systemctl enable --now docker
snap install amazon-ssm-agent --classic || true
systemctl enable --now snap.amazon-ssm-agent.amazon-ssm-agent.service || systemctl enable --now amazon-ssm-agent || true

mkdir -p /opt/llmenhance/qdrant
mkdir -p /opt/llmenhance/storage
mkdir -p /opt/llmenhance/app

if [ ! -d /opt/llmenhance/app/.git ]; then
  rm -rf /opt/llmenhance/app
  git clone "${repo_url}" /opt/llmenhance/app
fi

cd /opt/llmenhance/app
git fetch --all --prune
git checkout "${repo_ref}"

cat > /opt/llmenhance/app/.env <<'ENVEOF'
TEAM_ENV_PROFILE=aws-app-ec2
OLLAMA_BASE_URL=${ollama_base_url}
LLM_MODEL=qwen3:4b-instruct
EMBEDDING_MODEL=bge-m3

QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=llmenhance_chunks

RETRIEVAL_TOP_K=5
TEMPERATURE=0.2
NUM_CTX=4096
NUM_PREDICT=512

ENABLE_GEMINI_ENDPOINT=false
ENABLE_GEMINI_PANEL=false
ENVEOF

docker compose -f docker-compose.aws.yml up -d --build
```

- [ ] **Step 2: Check template formatting**

```powershell
terraform fmt -recursive infra/terraform
git diff --check
```

Expected:

```text
No formatting or whitespace errors.
```

- [ ] **Step 3: Commit**

```bash
git add infra/terraform/envs/mvp/user_data_app.sh.tftpl
git commit -m "infra: bootstrap app EC2 with docker compose"
```

---

### Task 5: Deploy And Verify The MVP

**Files:**
- No repository files if Tasks 1-4 are complete.

- [ ] **Step 1: Apply Terraform**

```powershell
cd infra/terraform/envs/mvp
terraform plan
terraform apply
terraform output app_instance_id
```

Expected:

```text
Apply complete!
app_instance_id prints the created EC2 instance id.
```

- [ ] **Step 2: Confirm the existing Ollama/Qwen model server is reachable from the app EC2**

If the existing model server uses a public security-group allowlist, add the app EIP as a `/32` source for TCP `11434` before testing:

```powershell
terraform output -raw app_public_ip
```

For example, if the output is `203.0.113.10`, the model server inbound rule should allow only `203.0.113.10/32` to TCP `11434` for this app deployment. If the model server is reachable through a private VPC route, VPN, or peering path, use that private path instead of public ingress.

Use SSM:

```powershell
$instanceId = terraform output -raw app_instance_id
aws ssm start-session --target $instanceId
```

Run inside the EC2 session:

```bash
source /opt/llmenhance/app/.env
curl -fsS "$OLLAMA_BASE_URL/api/tags" | python3 -m json.tool
```

Expected:

```text
The response includes qwen3:4b-instruct and bge-m3 or bge-m3:latest.
```

- [ ] **Step 3: Build the Qdrant index**

Run inside the EC2 session:

```bash
cd /opt/llmenhance/app
docker compose -f docker-compose.aws.yml run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
```

Expected:

```text
Documents indexed: 1
Chunks created: a positive number
Vectors inserted: a positive number
```

- [ ] **Step 4: Run EC2-local verification**

Run inside the EC2 session:

```bash
cd /opt/llmenhance/app
chmod +x scripts/aws_verify.sh
./scripts/aws_verify.sh
```

Expected:

```text
healthcheck succeeds.
The sample /api/ask/qwen response contains answer, sources, and elapsed_ms.
sources is not empty.
```

- [ ] **Step 5: Open a local Streamlit tunnel**

Run from the operator workstation:

```powershell
cd infra/terraform/envs/mvp
terraform output -raw ssm_streamlit_tunnel_command
```

Run the printed command, then open:

```text
http://localhost:8501
```

- [ ] **Step 6: Verify the employee-policy question path**

Ask:

```text
출장비 정산은 언제까지 해야 하나요?
```

Expected:

```text
The Qwen answer is grounded in retrieved policy context.
The response includes at least one source reference from datasets/docs/regulations.md.
If the UI still shows Gemini errors, keep the deployment internal and complete Task 6 before employee-facing exposure.
```

---

### Task 6: Remove Gemini From The Product Deployment Surface

**Files:**
- Modify: `app/server.py`
- Modify: `frontend/streamlit_app.py`
- Modify: `tests/test_server.py`
- Modify: `docs/AWS_DEPLOYMENT.md`
- Modify: `.env.aws-ec2.example`

The current main branch includes a Gemini comparison panel. That is useful for internal speed comparison, but the product deployment must default to Qwen-only.

- [ ] **Step 1: Add a shared env flag helper to `app/server.py`**

Add near the existing constants:

```python
def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _gemini_endpoint_enabled() -> bool:
    return _env_flag("ENABLE_GEMINI_ENDPOINT", default=False)
```

- [ ] **Step 2: Gate the Gemini API route**

Add this guard at the top of `ask_gemini`:

```python
@app.post("/api/ask/gemini", response_model=AskResponse)
def ask_gemini(req: AskRequest) -> AskResponse:
    if not _gemini_endpoint_enabled():
        raise HTTPException(status_code=404, detail="Gemini endpoint is disabled.")

    settings = Settings.from_env()
```

- [ ] **Step 3: Update server tests**

Add:

```python
def test_ask_gemini_is_disabled_by_default(monkeypatch):
    server = server_module()

    monkeypatch.delenv("ENABLE_GEMINI_ENDPOINT", raising=False)

    with pytest.raises(Exception) as exc_info:
        server.ask_gemini(server.AskRequest(question="재택근무 승인 절차는 어떻게 되나요?"))

    assert getattr(exc_info.value, "status_code", None) == 404
```

Then update existing Gemini tests so they explicitly enable the experiment path:

```python
monkeypatch.setenv("ENABLE_GEMINI_ENDPOINT", "true")
```

- [ ] **Step 4: Add an environment flag to `frontend/streamlit_app.py`**

Add:

```python
def _gemini_enabled() -> bool:
    return os.getenv("ENABLE_GEMINI_PANEL", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
```

- [ ] **Step 5: Hide Gemini service status when the panel is disabled**

In `_fetch_service_status`, return Gemini as disabled instead of an error:

```python
def _fetch_service_status() -> dict[str, Any]:
    try:
        response = httpx.get(HEALTH_ENDPOINT, timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        status = response.json()
        if not _gemini_enabled():
            status["gemini"] = {"status": "ok", "detail": "Gemini panel disabled."}
        return status
    except Exception as exc:
        gemini_status = {"status": "unknown", "detail": ""}
        if not _gemini_enabled():
            gemini_status = {"status": "ok", "detail": "Gemini panel disabled."}
        return {
            "api": {"status": "error", "detail": f"API 서버 연결 실패: {exc}"},
            "ollama": {"status": "unknown", "detail": ""},
            "qdrant": {"status": "unknown", "detail": ""},
            "gemini": gemini_status,
        }
```

In `_render_sidebar`, render the Gemini status line only when enabled:

```python
        _render_status_line("API 서버", status.get("api", {}))
        _render_status_line("Ollama / Qwen", status.get("ollama", {}))
        _render_status_line("Qdrant", status.get("qdrant", {}))
        if _gemini_enabled():
            _render_status_line("Vertex Gemini", status.get("gemini", {}))
```

- [ ] **Step 6: Use the flag in `_render_main`**

Change the first part of `_render_main` to:

```python
def _render_main() -> None:
    st.title("사내 규정 챗봇")

    status = st.session_state.service_status
    if not _gemini_enabled():
        _render_panel_header("EC2 Ollama / Qwen", status, "ollama")
        st.caption("온프레미스 RAG 답변 생성")
        st.divider()
        _render_chat_history(st.session_state.qwen_messages)

        question = st.chat_input("사내 규정에 대해 질문하세요.")
        if not question:
            return

        st.session_state.qwen_messages.append({"role": "user", "content": question})
        with st.spinner("문서 근거를 검색하고 답변을 생성하는 중..."):
            result = _call_rag(QWEN_ENDPOINT, {"question": question})
        _append_assistant_message("qwen_messages", result)
        st.rerun()

    col_qwen, col_gemini = st.columns(2)
```

Keep the existing two-column comparison code after the new guard.

- [ ] **Step 7: Document the default**

Add to `docs/AWS_DEPLOYMENT.md`:

````markdown
## Product Runtime

The AWS MVP deployment uses the Qwen-only Streamlit surface by default:

```env
ENABLE_GEMINI_PANEL=false
ENABLE_GEMINI_ENDPOINT=false
```

The Gemini comparison panel is allowed only for internal benchmark sessions.
````

- [ ] **Step 8: Add the env to `.env.aws-ec2.example`**

```env
ENABLE_GEMINI_PANEL=false
ENABLE_GEMINI_ENDPOINT=false
```

- [ ] **Step 9: Run tests and local compose config**

```powershell
docker compose run --rm rag-api pytest -v
docker compose -f docker-compose.aws.yml config
```

Expected:

```text
pytest passes.
docker compose config succeeds.
```

- [ ] **Step 10: Commit**

```bash
git add app/server.py frontend/streamlit_app.py tests/test_server.py docs/AWS_DEPLOYMENT.md .env.aws-ec2.example
git commit -m "feat: default AWS deployment to Qwen-only UI"
```

---

### Task 7: Add Operations, Backup, And Promotion Gates

**Files:**
- Modify: `docs/AWS_DEPLOYMENT.md`
- Optional create: `infra/terraform/envs/mvp/operations.tf`

- [ ] **Step 1: Add backup notes to `docs/AWS_DEPLOYMENT.md`**

```markdown
## Backup

- Snapshot the app EC2 root EBS volume before any destructive re-indexing.
- Use AWS Data Lifecycle Manager for daily EBS snapshots during MVP testing.
- Keep at least seven daily snapshots.
- Do not commit Qdrant data, SQLite files, `.env`, or Terraform state files.
```

- [ ] **Step 2: Add logging notes**

```markdown
## Logging

- CloudWatch Agent may collect host metrics, Docker logs, disk usage, and memory usage.
- Application logs must not include full confidential document chunks by default.
- The sample RAG answer can be logged only with answer metadata and source ids.
```

- [ ] **Step 3: Add rollback procedure**

````markdown
## Rollback

```bash
cd /opt/llmenhance/app
git fetch --all --prune
git checkout 05e4597ad5bca41c030ad6645ee1a6e19f16d443
docker compose -f docker-compose.aws.yml up -d --build
./scripts/aws_verify.sh
```
````

- [ ] **Step 4: Add public exposure gates**

```markdown
## Public Exposure Gates

Do not expose Streamlit or `/api/ask/qwen` to the public internet until all are complete:

- Authentication exists.
- HTTPS exists.
- CORS is restricted to the real frontend origin.
- Request body size and rate limits exist.
- Port `11434` remains private.
- Port `6333` remains private.
- Product UI is Qwen-only by default.
```

- [ ] **Step 5: Commit**

```bash
git add docs/AWS_DEPLOYMENT.md
git commit -m "docs: add AWS operations and exposure gates"
```

---

## Review Notes

Review conclusions before execution:

- The safest Terraform boundary is app-side AWS resources only. The manually prepared Ollama/Qwen EC2 should remain an external model server for the first release.
- The app EC2 can be CPU-based because Qwen and embedding inference happen on the existing model server.
- The Terraform MVP stack opens no inbound app ports; SSM port forwarding is enough for MVP validation.
- Current main has Gemini comparison UI and API routes. Employee-facing AWS deployment must complete Task 6 or stay internal.
- Current main has wildcard CORS. Public ALB/HTTPS deployment should wait for auth and restricted CORS.
- `ollama_base_url` is required and has no public hard-coded default. Prefer a private model-server route; otherwise allowlist the app EIP as a `/32`.
- `repo_ref` defaults to `main`; when deploying a branch before merge, set it to a pushed branch or commit that contains the AWS runtime files.
- `user_data_replace_on_change = true` is included so Terraform can replace the app EC2 when deployment bootstrap inputs change. For later zero-downtime releases, move promotion to SSM Run Command or image-based deployment.
- The verification script must fail when the grounded sample answer has empty `sources`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-21-aws-deployment-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, with checkpoints.

Recommended split:

- Runtime files: Tasks 1 and 6.
- Terraform files: Tasks 2, 3, and 4.
- EC2 deployment verification: Task 5.
- Operations docs: Task 7.
