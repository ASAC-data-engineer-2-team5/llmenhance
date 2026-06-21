param(
    [ValidateSet("shared-ec2", "local-ollama")]
    [string]$Profile = "shared-ec2",
    [switch]$ForceEnv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Require-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is required. $InstallHint"
    }
}

function Read-DotEnv {
    param([string]$Path)

    $values = @{}
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) {
            return
        }

        $parts = $line -split "=", 2
        if ($parts.Count -eq 2) {
            $values[$parts[0]] = $parts[1]
        }
    }

    return $values
}

function Test-OllamaEndpoint {
    param([string]$BaseUrl)

    $tagsUrl = "$BaseUrl/api/tags"
    try {
        return Invoke-RestMethod -Uri $tagsUrl -TimeoutSec 20
    } catch {
        throw "Ollama endpoint is not reachable at $tagsUrl. Check VPN/security group/IP allowlist, then rerun setup."
    }
}

function Assert-OllamaModels {
    param(
        [object]$Tags,
        [string]$LlmModel,
        [string]$EmbeddingModel
    )

    $modelNames = @($Tags.models | ForEach-Object { $_.name })
    if ($modelNames -notcontains $LlmModel) {
        throw "Required LLM model $LlmModel was not found on the configured Ollama endpoint."
    }

    $embeddingLatestName = "${EmbeddingModel}:latest"
    if (($modelNames -notcontains $EmbeddingModel) -and ($modelNames -notcontains $embeddingLatestName)) {
        throw "Required embedding model $EmbeddingModel was not found on the configured Ollama endpoint."
    }
}

Write-Step "Checking required local tools"
Require-Command "docker" "Install and start Docker Desktop, then rerun this script."

docker version | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Docker is installed but not reachable. Start Docker Desktop, then rerun this script."
}

docker compose version | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose is not available through the Docker CLI."
}

$knownTemplates = @(".env.shared-ec2.example", ".env.local-ollama.example")
$templatePath = ".env.$Profile.example"
if (-not (Test-Path $templatePath)) {
    throw "Environment template not found: $templatePath. Expected one of: $($knownTemplates -join ', ')"
}

Write-Step "Creating .env from $templatePath"
if ((Test-Path ".env") -and $ForceEnv) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupPath = ".env.backup.$timestamp"
    Copy-Item .env $backupPath
    Write-Host "Backed up existing .env to $backupPath"
}

if ((-not (Test-Path ".env")) -or $ForceEnv) {
    Copy-Item $templatePath .env
    Write-Host "Created .env for profile $Profile"
} else {
    Write-Host ".env already exists. Rerun with -ForceEnv to replace it with profile $Profile."
}

$envValues = Read-DotEnv ".env"
$ollamaBaseUrl = $envValues["OLLAMA_BASE_URL"]
$llmModel = $envValues["LLM_MODEL"]
$embeddingModel = $envValues["EMBEDDING_MODEL"]

if ($Profile -eq "local-ollama") {
    Write-Step "Checking local Ollama and pulling required models"
    Require-Command "ollama" "Install Ollama on the Windows host, then rerun this script."

    ollama list | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Ollama is installed but not reachable. Start Ollama on the Windows host."
    }

    ollama pull $embeddingModel
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to pull $embeddingModel."
    }

    ollama pull $llmModel
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to pull $llmModel."
    }
} else {
    Write-Step "Checking shared EC2 Ollama endpoint"
    $tags = Test-OllamaEndpoint $ollamaBaseUrl
    Assert-OllamaModels $tags $llmModel $embeddingModel
    Write-Host "Shared Ollama endpoint is reachable: $ollamaBaseUrl"
}

Write-Step "Building and starting Docker services"
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    throw "docker compose up failed."
}

Write-Step "Rebuilding the Qdrant index from datasets/docs"
docker compose run --rm rag-api python scripts/ingest_md.py datasets/docs --reset
if ($LASTEXITCODE -ne 0) {
    throw "Document ingestion failed."
}

Write-Host ""
Write-Host "SETUP_DONE"
Write-Host "Next: run scripts/dev_verify.ps1"
