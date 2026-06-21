Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message"
}

function Require-Success {
    param(
        [int]$ExitCode,
        [string]$FailureMessage
    )

    if ($ExitCode -ne 0) {
        throw $FailureMessage
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

function Assert-OllamaEndpoint {
    param(
        [string]$BaseUrl,
        [string]$LlmModel,
        [string]$EmbeddingModel
    )

    $tags = Invoke-RestMethod -Uri "$BaseUrl/api/tags" -TimeoutSec 20
    $modelNames = @($tags.models | ForEach-Object { $_.name })
    if ($modelNames -notcontains $LlmModel) {
        throw "Required LLM model $LlmModel was not found at $BaseUrl."
    }

    $embeddingLatestName = "${EmbeddingModel}:latest"
    if (($modelNames -notcontains $EmbeddingModel) -and ($modelNames -notcontains $embeddingLatestName)) {
        throw "Required embedding model $EmbeddingModel was not found at $BaseUrl."
    }
}

Write-Step "Checking active environment profile"
if (-not (Test-Path ".env")) {
    throw ".env does not exist. Run scripts/dev_setup.ps1 first."
}

$envValues = Read-DotEnv ".env"
$profile = $envValues["TEAM_ENV_PROFILE"]
$ollamaBaseUrl = $envValues["OLLAMA_BASE_URL"]
$llmModel = $envValues["LLM_MODEL"]
$embeddingModel = $envValues["EMBEDDING_MODEL"]

Write-Host "TEAM_ENV_PROFILE=$profile"
Write-Host "OLLAMA_BASE_URL=$ollamaBaseUrl"
Write-Host "LLM_MODEL=$llmModel"
Write-Host "EMBEDDING_MODEL=$embeddingModel"

Assert-OllamaEndpoint $ollamaBaseUrl $llmModel $embeddingModel

Write-Step "Checking Docker Compose services"
docker compose ps
Require-Success $LASTEXITCODE "docker compose ps failed."

Write-Step "Checking Qdrant HTTP endpoint"
curl.exe http://localhost:6333
Require-Success $LASTEXITCODE "Qdrant did not respond at http://localhost:6333."

Write-Step "Running app healthcheck"
docker compose run --rm rag-api python -m app.healthcheck
Require-Success $LASTEXITCODE "app.healthcheck failed."

Write-Step "Running test suite"
docker compose run --rm rag-api pytest -v
Require-Success $LASTEXITCODE "pytest failed."

Write-Step "Running sample grounded RAG question"
$sampleQuestionCodes = @(
    0xBC95, 0xC778, 0xCE74, 0xB4DC, 0x0020,
    0xC0AC, 0xC6A9, 0x0020,
    0xD6C4, 0x0020,
    0xC804, 0xD45C, 0x0020,
    0xCC98, 0xB9AC, 0xB294, 0x0020,
    0xC5B8, 0xC81C, 0xAE4C, 0xC9C0, 0x0020,
    0xD574, 0xC57C, 0x0020,
    0xD558, 0xB098, 0xC694, 0x003F
)
$fallbackPhraseCodes = @(
    0xBB38, 0xC11C, 0xC5D0, 0xC11C, 0x0020,
    0xD655, 0xC778, 0xB418, 0xC9C0, 0x0020,
    0xC54A, 0xC2B5, 0xB2C8, 0xB2E4
)
$sampleQuestion = -join ($sampleQuestionCodes | ForEach-Object { [char]$_ })
$fallbackPhrase = -join ($fallbackPhraseCodes | ForEach-Object { [char]$_ })
$sampleExitCode = 1
$ragOutput = @()
$process = $null
try {
    $processStartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $processStartInfo.FileName = "docker"
    $processStartInfo.Arguments = "compose run --rm rag-api python scripts/ask_rag.py `"$sampleQuestion`" --top-k 3 --timing"
    $processStartInfo.RedirectStandardOutput = $true
    $processStartInfo.RedirectStandardError = $true
    $processStartInfo.UseShellExecute = $false

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $processStartInfo
    [void]$process.Start()
    $ragStdout = $process.StandardOutput.ReadToEnd()
    $ragStderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    $sampleExitCode = $process.ExitCode

    if ($ragStdout) {
        $ragOutput = @($ragOutput) + @($ragStdout -split "`r?`n")
    }
    if ($ragStderr) {
        $ragOutput = @($ragOutput) + @($ragStderr -split "`r?`n")
    }
} finally {
    if ($process) {
        $process.Dispose()
    }
}
$ragOutput = @($ragOutput | Where-Object { $_ })
$ragText = $ragOutput -join "`n"

Require-Success $sampleExitCode "Sample RAG question failed."
$ragOutput | Out-Host

if ($ragText -notmatch "Sources:") {
    throw "Sample RAG output did not include Sources:"
}

if ($ragText -match "- none") {
    throw "Sample RAG output returned no sources."
}

if ($ragText.Contains($fallbackPhrase)) {
    throw "Sample RAG output fell back instead of answering from retrieved context."
}

Write-Host ""
Write-Host "SETUP_OK"
