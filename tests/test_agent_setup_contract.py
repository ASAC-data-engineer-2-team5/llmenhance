import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def decode_powershell_char_codes(script: str, variable_name: str) -> str:
    match = re.search(rf"\${variable_name}\s*=\s*@\((.*?)\)", script, re.DOTALL)
    assert match is not None
    codes = re.findall(r"0x[0-9A-Fa-f]+", match.group(1))
    return "".join(chr(int(code, 16)) for code in codes)


def test_readme_has_agent_setup_quickstart_contract() -> None:
    readme = read_repo_file("README.md")

    assert "## Agent Setup Quickstart" in readme
    assert "CLI coding agent" in readme
    assert "shared EC2 Ollama endpoint" in readme
    assert ".\\scripts\\dev_setup.ps1 -Profile shared-ec2 -ForceEnv" in readme
    assert "scripts/dev_verify.ps1" in readme
    assert "SETUP_OK" in readme
    assert "Do not move Ollama/Qwen into Docker" in readme
    assert "local-ollama" in readme
    assert "docs/TEAM_ENVIRONMENT.md" in readme


def test_dev_setup_script_bootstraps_profile_driven_environment() -> None:
    script = read_repo_file("scripts/dev_setup.ps1")

    assert '[ValidateSet("shared-ec2", "local-ollama")]' in script
    assert '[string]$Profile = "shared-ec2"' in script
    assert "[switch]$ForceEnv" in script
    assert ".env.shared-ec2.example" in script
    assert ".env.local-ollama.example" in script
    assert "Test-OllamaEndpoint" in script
    assert "Assert-OllamaModels" in script
    assert 'if ($Profile -eq "local-ollama")' in script
    assert 'Require-Command "ollama"' in script
    assert "ollama pull $embeddingModel" in script
    assert "ollama pull $llmModel" in script
    assert "docker compose up -d --build" in script
    assert "python scripts/ingest_md.py datasets/docs --reset" in script
    assert "SETUP_DONE" in script


def test_dev_setup_can_backup_existing_env_when_forcing_profile() -> None:
    script = read_repo_file("scripts/dev_setup.ps1")

    assert ".env.backup." in script
    assert "Copy-Item .env $backupPath" in script
    assert "Copy-Item $templatePath .env" in script


def test_shared_ec2_env_template_uses_team_endpoint() -> None:
    env_template = read_repo_file(".env.shared-ec2.example")

    assert "TEAM_ENV_PROFILE=shared-ec2" in env_template
    assert "OLLAMA_BASE_URL=http://16.208.81.115:11434" in env_template
    assert "LLM_MODEL=qwen3:4b-instruct" in env_template
    assert "EMBEDDING_MODEL=bge-m3" in env_template
    assert "QDRANT_URL=http://qdrant:6333" in env_template


def test_local_ollama_env_template_preserves_on_prem_story() -> None:
    env_template = read_repo_file(".env.local-ollama.example")

    assert "TEAM_ENV_PROFILE=local-ollama" in env_template
    assert "OLLAMA_BASE_URL=http://host.docker.internal:11434" in env_template
    assert "LLM_MODEL=qwen3:4b-instruct" in env_template
    assert "EMBEDDING_MODEL=bge-m3" in env_template


def test_env_example_uses_local_ollama_bootstrap_models() -> None:
    env_example = read_repo_file(".env.example")

    assert "TEAM_ENV_PROFILE=local-ollama" in env_example
    assert "LLM_MODEL=qwen3:4b-instruct" in env_example
    assert "EMBEDDING_MODEL=bge-m3" in env_example
    assert "OLLAMA_BASE_URL=http://host.docker.internal:11434" in env_example


def test_dev_verify_script_checks_runtime_and_prints_setup_ok() -> None:
    script = read_repo_file("scripts/dev_verify.ps1")

    assert "Read-DotEnv" in script
    assert "TEAM_ENV_PROFILE" in script
    assert "OLLAMA_BASE_URL" in script
    assert "/api/tags" in script
    assert "qwen3:4b-instruct" in read_repo_file(".env.shared-ec2.example")
    assert "bge-m3" in read_repo_file(".env.shared-ec2.example")
    assert "http://localhost:6333" in script
    assert "python -m app.healthcheck" in script
    assert "pytest -v" in script
    assert "python scripts/ask_rag.py" in script
    assert "$sampleQuestionCodes" in script
    assert "0xBC95" in script
    assert "$sampleQuestion = -join" in script
    assert "Sources:" in script
    assert "$fallbackPhraseCodes" in script
    assert "$fallbackPhrase = -join" in script
    assert "$ragText.Contains($fallbackPhrase)" in script
    assert "SETUP_OK" in script


def test_dev_verify_sample_question_is_confirmed_by_current_corpus() -> None:
    script = read_repo_file("scripts/dev_verify.ps1")

    assert decode_powershell_char_codes(script, "sampleQuestionCodes") == (
        "법인카드 사용 후 전표 처리는 언제까지 해야 하나요?"
    )


def test_dev_verify_allows_docker_status_stderr_during_sample_question() -> None:
    script = read_repo_file("scripts/dev_verify.ps1")

    assert "System.Diagnostics.ProcessStartInfo" in script
    assert 'FileName = "docker"' in script
    assert "RedirectStandardOutput = $true" in script
    assert "RedirectStandardError = $true" in script
    assert "$sampleExitCode = $process.ExitCode" in script


def test_gitignore_excludes_local_rag_state() -> None:
    gitignore = read_repo_file(".gitignore")

    assert "storage/" in gitignore
    assert "*.sqlite" in gitignore
    assert "*.sqlite3" in gitignore
    assert ".env.backup.*" in gitignore
    assert "!.env.shared-ec2.example" in gitignore
    assert "!.env.local-ollama.example" in gitignore


def test_team_environment_doc_exists_and_mentions_security_group() -> None:
    doc = read_repo_file("docs/TEAM_ENVIRONMENT.md")

    assert "shared-ec2" in doc
    assert "local-ollama" in doc
    assert "http://16.208.81.115:11434" in doc
    assert "security group" in doc
    assert "SETUP_OK" in doc


def test_existing_ec2_compose_matches_current_app_runtime_contract() -> None:
    compose = read_repo_file("docker-compose.aws.yml")

    assert "streamlit run frontend/streamlit_app.py" in compose
    assert "RAG_API_URL: http://rag-api:8000" in compose
    assert "/opt/llmenhance/qdrant:/qdrant/storage" in compose
    assert "/opt/llmenhance/storage:/app/storage" in compose
    assert "GOOGLE_APPLICATION_CREDENTIALS_HOST_PATH" in compose
    assert "/run/secrets/google-application-credentials.json:ro" in compose
    assert "host.docker.internal:host-gateway" in compose
    assert "127.0.0.1:8000:8000" in compose
    assert "0.0.0.0:8501:8501" in compose


def test_existing_ec2_env_template_includes_model_comparison_settings() -> None:
    env_template = read_repo_file(".env.aws-ec2.example")

    assert "TEAM_ENV_PROFILE=existing-ec2" in env_template
    assert "OLLAMA_BASE_URL=http://host.docker.internal:11434" in env_template
    assert "LLM_MODEL=qwen3:4b-instruct" in env_template
    assert "OLLAMA_EMBEDDING_TIMEOUT_SECONDS=180" in env_template
    assert "TEMPERATURE=0.2" in env_template
    assert "NUM_CTX=4096" in env_template
    assert "NUM_PREDICT=512" in env_template
    assert (
        "GOOGLE_APPLICATION_CREDENTIALS_HOST_PATH=/home/ubuntu/secrets/"
        "google-application-credentials.json"
    ) in env_template
    assert "GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/google-application-credentials.json" in (
        env_template
    )
    assert "GOOGLE_CLOUD_PROJECT=" in env_template
    assert "GOOGLE_CLOUD_LOCATION=us-central1" in env_template
    assert "GEMINI_MODEL=gemini-2.5-flash" in env_template
    assert "GEMINI_THINKING_BUDGET=0" in env_template
    assert "BEDROCK_REGION=ap-northeast-2" in env_template
    assert "PRESENTATION_TOP_K=3" in env_template


def test_existing_ec2_deploy_workflow_targets_existing_instance_and_reindexes() -> None:
    workflow = read_repo_file(".github/workflows/deploy-existing-ec2.yml")

    assert "workflow_dispatch:" in workflow
    assert "git_ref:" in workflow
    assert 'default: "main"' in workflow
    assert "AWS_APP_INSTANCE_ID" in workflow
    assert "i-0ccf9071972894f30" in workflow
    assert "aws-actions/configure-aws-credentials@v4" in workflow
    assert "aws ssm send-command" in workflow
    assert "/opt/llmenhance/app" in workflow
    assert "git fetch --all --prune" in workflow
    assert "OLLAMA_EMBEDDING_TIMEOUT_SECONDS=180" in workflow
    assert "git pull --ff-only origin" in workflow
    assert "docker compose -f docker-compose.aws.yml up -d --build" in workflow
    assert "python scripts/ingest_md.py datasets/docs --reset" in workflow
    assert "./scripts/aws_verify.sh" in workflow


def test_aws_verify_checks_existing_ec2_runtime_services() -> None:
    script = read_repo_file("scripts/aws_verify.sh")

    assert "http://127.0.0.1:8501" in script
    assert "http://127.0.0.1:8000/health/services" in script
    assert "http://127.0.0.1:6333/collections/llmenhance_chunks" in script
    assert "points_count" in script
    assert "AWS_VERIFY_OK" in script
