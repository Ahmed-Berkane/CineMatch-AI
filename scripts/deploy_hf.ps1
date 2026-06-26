# Deploy CineMatch-AI to Hugging Face Spaces (Streamlit).
# Usage (from repo root, with venv activated):
#   hf auth login
#   .\scripts\deploy_hf.ps1
# Optional:
#   .\scripts\deploy_hf.ps1 -SpaceName "CineMatch-AI" -Username "YourHfUsername"

param(
    [string]$SpaceName = "CineMatch-AI",
    [string]$Username = "",
    [string]$Token = "",
    [switch]$SkipCreate,
    [switch]$Full,
    [switch]$FreeSpace
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
# Hidden staging folder (gitignored). Uploaded via HF Hub API — no git push / credentials needed.
$HfDir = Join-Path $Root ".hf-staging"

function Require-Command($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Missing command: $name"
    }
}

function Get-HfCli {
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if ($hf) { return $hf.Source }

    $venvHf = Join-Path $Root "venv/Scripts/hf.exe"
    if (Test-Path $venvHf) { return $venvHf }

    throw "Missing hf CLI. Install with: pip install -U huggingface_hub`nThen run: hf auth login"
}

function Invoke-Hf {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & (Get-HfCli) @Args
    if ($LASTEXITCODE -ne 0) {
        throw "hf command failed: hf $($Args -join ' ')"
    }
}

function Upload-HfSpace {
    param(
        [string]$RepoId,
        [string]$Folder,
        [string]$Message,
        [string]$HfToken,
        [ValidateSet("code", "model", "all")]
        [string]$Mode = "code"
    )
    Require-Command python
    $uploadScript = Join-Path $Root "scripts/upload_hf_space.py"
    if (-not (Test-Path $uploadScript)) {
        throw "Missing $uploadScript"
    }
    $pyArgs = @(
        $uploadScript,
        "--repo-id", $RepoId,
        "--folder", $Folder,
        "--message", $Message
    )
    if ($Mode -eq "code") {
        $pyArgs += "--code-only"
    } elseif ($Mode -eq "model") {
        $pyArgs += "--with-model"
    }
    if ($HfToken) {
        $pyArgs += @("--token", $HfToken)
    }
    & python @pyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "HF upload failed"
    }
}

Require-Command python
$HfCli = Get-HfCli

Write-Host "Using HF CLI: $HfCli" -ForegroundColor Gray
Write-Host "Checking Hugging Face login..." -ForegroundColor Cyan

try {
    $who = Invoke-Hf auth whoami --format json | ConvertFrom-Json
} catch {
    throw "Not logged in. Run: hf auth login`nGet a token at https://huggingface.co/settings/tokens (Write access)."
}

if (-not $Username) {
    $Username = $who.user
    if (-not $Username) { $Username = $who.name }
}
if (-not $Username) {
    throw "Could not detect HF username. Pass -Username YourHfUsername"
}

if ($Token) {
    $hfToken = $Token.Trim()
    Write-Host "Using provided HF token..." -ForegroundColor Cyan
    Invoke-Hf auth login --token $hfToken --force | Out-Null
} else {
    $hfToken = (Invoke-Hf auth token -q).Trim()
}

$SpaceId = "$Username/$SpaceName"
$RemoteUrl = "https://huggingface.co/spaces/$SpaceId"
Write-Host "Target Space: $RemoteUrl" -ForegroundColor Green

try {
    $spaceInfo = Invoke-Hf api spaces/$SpaceId 2>$null | ConvertFrom-Json
    if ($spaceInfo.usedStorage -and $spaceInfo.usedStorage -gt 1000000000) {
        $usedGb = [math]::Round($spaceInfo.usedStorage / 1GB, 2)
        Write-Host "WARNING: Space storage is ${usedGb} GB (limit 1 GB). Upload may fail." -ForegroundColor Yellow
        Write-Host "Free orphaned LFS blobs (e.g. old train/val/test parquets):" -ForegroundColor Yellow
        Write-Host "  python scripts/free_hf_lfs.py --repo-id $SpaceId" -ForegroundColor Yellow
        Write-Host "Or: Space Settings -> Storage -> List LFS files -> delete old large objects." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not check Space storage quota." -ForegroundColor Gray
}

if ($FreeSpace) {
    Write-Host "Freeing Space storage (removing split parquets - app uses movies_catalog.parquet)..." -ForegroundColor Cyan
    Invoke-Hf repos delete-files $SpaceId `
        "data/processed/val.parquet" `
        "data/processed/test.parquet" `
        "data/processed/train.parquet" `
        --repo-type space `
        --commit-message "Remove split parquets; serve catalog from artifacts/movies_catalog.parquet"
}

if (-not $SkipCreate) {
    Write-Host "Creating Space (ignored if it already exists)..." -ForegroundColor Cyan
    Invoke-Hf repos create $SpaceName --type space --space-sdk docker --exist-ok
}

if (Test-Path $HfDir) {
    Remove-Item $HfDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $HfDir | Out-Null
Write-Host "Staging files in $HfDir ..." -ForegroundColor Cyan

$ExcludeDirs = @(".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", "Notebooks", "data", "scripts", "artifacts")
$ExcludeFiles = @(
    ".env", ".env.local", ".env.example",
    "Project.docx",
    "HUGGINGFACE.md", "README.md", ".gitignore"
)
$ExcludePatterns = @("~$*")

$RootFiles = @("app.py", "Dockerfile", ".dockerignore", "requirements.txt", "Logo.png")
$RuntimeScripts = @(
    "__init__.py",
    "catalog.py",
    "data_helpers.py",
    "explainability.py",
    "feedback.py",
    "model_helpers.py",
    "neural_models.py",
    "persona.py",
    "recommender.py",
    "taste_profile.py",
    "taste_utils.py"
)

Write-Host "Syncing project files..." -ForegroundColor Cyan
Get-ChildItem $Root -Force | ForEach-Object {
    $name = $_.Name
    if ($ExcludeDirs -contains $name) { return }
    if ($ExcludeFiles -contains $name) { return }
    foreach ($pat in $ExcludePatterns) {
        if ($name -like $pat) { return }
    }
    if ($name -eq ".streamlit") {
        $dest = Join-Path $HfDir ".streamlit"
        if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
        Copy-Item $_.FullName $dest -Recurse -Force
        return
    }
    if ($RootFiles -notcontains $name) { return }
    Copy-Item $_.FullName (Join-Path $HfDir $name) -Force
}

$scriptsDest = Join-Path $HfDir "scripts"
if (Test-Path $scriptsDest) { Remove-Item $scriptsDest -Recurse -Force }
New-Item -ItemType Directory -Force -Path $scriptsDest | Out-Null
foreach ($script in $RuntimeScripts) {
    $src = Join-Path $Root "scripts/$script"
    if (-not (Test-Path $src)) {
        throw "Missing runtime script: $src"
    }
    Copy-Item $src (Join-Path $scriptsDest $script) -Force
}

$destArtifacts = Join-Path $HfDir "artifacts"

function Stage-CatalogParquet {
    param([string]$DestDir)
    $catalog = Join-Path $Root "artifacts/movies_catalog.parquet"
    if (-not (Test-Path $catalog)) { return }
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    $dest = Join-Path $DestDir "movies_catalog.parquet"
    Write-Host "Staging slim catalog for Space (drops overview to save LFS quota)..." -ForegroundColor Cyan
    & python (Join-Path $Root "scripts/slim_catalog.py") $catalog $dest
    if ($LASTEXITCODE -ne 0) { throw "slim_catalog.py failed" }
}

if ($Full) {
    if (Test-Path $destArtifacts) { Remove-Item $destArtifacts -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $destArtifacts | Out-Null
    $fullModel = Join-Path $Root "artifacts/best_model_full.pt"
    $valModel = Join-Path $Root "artifacts/best_model.pt"
    if (Test-Path $fullModel) {
        Write-Host "Using full-dataset checkpoint: best_model_full.pt" -ForegroundColor Green
        Copy-Item $fullModel (Join-Path $destArtifacts "best_model_full.pt") -Force
    } elseif (Test-Path $valModel) {
        Write-Host 'best_model_full.pt not found - falling back to best_model.pt (val-split winner).' -ForegroundColor Yellow
        Write-Host 'Run: python scripts/train_pipeline.py --retrain-best-full' -ForegroundColor Yellow
        Copy-Item $valModel (Join-Path $destArtifacts "best_model.pt") -Force
    } else {
        throw "Missing artifacts/best_model_full.pt or artifacts/best_model.pt. Train first: python scripts/train_pipeline.py"
    }
    Stage-CatalogParquet $destArtifacts
} else {
    Write-Host 'Code-only deploy: skipping model re-upload. Use -Full to include best_model_full.pt.' -ForegroundColor Gray
    if (Test-Path $destArtifacts) { Remove-Item $destArtifacts -Recurse -Force }
    Stage-CatalogParquet $destArtifacts
}

# Remove dev-only files that may linger from older deploys.
$RemoveFromSpace = @(
    ".env.example",
    "HUGGINGFACE.md",
    ".gitignore",
    "scripts/deploy_hf.ps1",
    "scripts/fetch_tmdb_metadata.py",
    "scripts/train_pipeline.py",
    "scripts/predict.py"
)
foreach ($rel in $RemoveFromSpace) {
    $target = Join-Path $HfDir $rel
    if (Test-Path $target) { Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue }
}

# Space README must include YAML frontmatter (see HUGGINGFACE.md).
$hfReadme = Join-Path $Root "HUGGINGFACE.md"
if (-not (Test-Path $hfReadme)) {
    throw "Missing HUGGINGFACE.md"
}
Copy-Item $hfReadme (Join-Path $HfDir "README.md") -Force

# Git LFS attributes (required by the Space repo for large binaries).
@"
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
"@ | Set-Content -Path (Join-Path $HfDir ".gitattributes") -Encoding utf8

$msg = "Deploy CineMatch-AI Streamlit app"
$uploadMode = if ($Full) { "model" } else { "code" }

if ($Full) {
    Write-Host "Freeing old Space LFS (splits + replaced checkpoints)..." -ForegroundColor Cyan
    $freeScript = Join-Path $Root "scripts/free_hf_lfs.py"
    & python $freeScript --repo-id $SpaceId --prefix data/processed/
    if ($LASTEXITCODE -ne 0) { throw "free_hf_lfs.py failed" }
    if (Test-Path (Join-Path $HfDir "artifacts/best_model_full.pt")) {
        & python $freeScript --repo-id $SpaceId --exact artifacts/best_model.pt
    }
}

Write-Host "Uploading to Hugging Face (large files may take several minutes)..." -ForegroundColor Cyan
try {
    Upload-HfSpace -RepoId $SpaceId -Folder $HfDir -Message $msg -HfToken $(if ($Token) { $hfToken } else { "" }) -Mode $uploadMode
} catch {
    throw @"
HF upload failed. Ensure you are logged in: hf auth login

If browser login fails, create a Write token at https://huggingface.co/settings/tokens then run:
  .\scripts\deploy_hf.ps1 -Username `"$Username`" -Token hf_xxxxxxxx

Original error: $_
"@
}

Write-Host "`nDone! Space URL: $RemoteUrl" -ForegroundColor Green
Write-Host "Build logs: $RemoteUrl (Logs tab). First cold start can take 1-2 minutes." -ForegroundColor Gray
