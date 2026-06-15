# Deploy CineMatch-AI to Hugging Face Spaces (Streamlit).
# Usage (from repo root, with venv activated):
#   hf auth login
#   .\scripts\deploy_hf.ps1
# Optional:
#   .\scripts\deploy_hf.ps1 -SpaceName "CineMatch-AI" -Username "YourHfUsername"

param(
    [string]$SpaceName = "CineMatch-AI",
    [string]$Username = "",
    [switch]$SkipCreate
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$HfDir = Join-Path (Split-Path -Parent $Root) "${SpaceName}-hf"

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

Require-Command git
Require-Command git-lfs
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

$SpaceId = "$Username/$SpaceName"
$RemoteUrl = "https://huggingface.co/spaces/$SpaceId"
Write-Host "Target Space: $RemoteUrl" -ForegroundColor Green

if (-not $SkipCreate) {
    Write-Host "Creating Space (ignored if it already exists)..." -ForegroundColor Cyan
    Invoke-Hf repos create $SpaceName --type space --space-sdk docker --exist-ok
}

if (-not (Test-Path $HfDir)) {
    Write-Host "Cloning Space repo to $HfDir ..." -ForegroundColor Cyan
    git clone $RemoteUrl $HfDir
} else {
    Write-Host "Using existing clone: $HfDir" -ForegroundColor Cyan
    Push-Location $HfDir
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    git pull origin main 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { git pull origin master 2>&1 | Out-Null }
    $ErrorActionPreference = $prevEap
    Pop-Location
}

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
if (Test-Path $destArtifacts) { Remove-Item $destArtifacts -Recurse -Force }
New-Item -ItemType Directory -Force -Path $destArtifacts | Out-Null
$model = Join-Path $Root "artifacts/best_model.pt"
if (-not (Test-Path $model)) {
    throw "Missing $model. Train the model first: python scripts/train_pipeline.py"
}
Copy-Item $model (Join-Path $destArtifacts "best_model.pt") -Force
$catalog = Join-Path $Root "artifacts/movies_catalog.parquet"
if (Test-Path $catalog) {
    Copy-Item $catalog (Join-Path $destArtifacts "movies_catalog.parquet") -Force
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

# Processed splits (required for search + cohort explanations).
$processedSrc = Join-Path $Root "data/processed"
$processedDest = Join-Path $HfDir "data/processed"
New-Item -ItemType Directory -Force -Path $processedDest | Out-Null
foreach ($f in @("train.parquet", "val.parquet", "test.parquet")) {
    $src = Join-Path $processedSrc $f
    if (-not (Test-Path $src)) {
        throw "Missing $src (required for the Space)."
    }
    Copy-Item $src (Join-Path $processedDest $f) -Force
}

# Space README must include YAML frontmatter (see HUGGINGFACE.md).
$hfReadme = Join-Path $Root "HUGGINGFACE.md"
if (-not (Test-Path $hfReadme)) {
    throw "Missing HUGGINGFACE.md"
}
Copy-Item $hfReadme (Join-Path $HfDir "README.md") -Force

# Git LFS for large binaries on the Space repo (HF rejects raw binaries without LFS/XET).
@"
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
"@ | Set-Content -Path (Join-Path $HfDir ".gitattributes") -Encoding utf8

Push-Location $HfDir
git lfs install | Out-Null
git lfs track "*.parquet" "*.pt" "*.png" | Out-Null

git add -A
git add -f artifacts/
foreach ($rel in $RemoveFromSpace) {
    git rm -f --ignore-unmatch $rel 2>&1 | Out-Null
}
$status = git status --porcelain
if (-not $status) {
    Write-Host "Nothing to commit - Space is already up to date." -ForegroundColor Yellow
    Pop-Location
    Write-Host "`nOpen: $RemoteUrl" -ForegroundColor Green
    exit 0
}

$msg = "Deploy CineMatch-AI Streamlit app"
git commit -m $msg
Write-Host "Pushing to Hugging Face (LFS upload may take several minutes)..." -ForegroundColor Cyan
git push
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    throw "git push failed. Check LFS files and binary rejections above."
}

Pop-Location
Write-Host "`nDone! Space URL: $RemoteUrl" -ForegroundColor Green
Write-Host "Build logs: $RemoteUrl (Logs tab). First cold start can take 1-2 minutes." -ForegroundColor Gray
