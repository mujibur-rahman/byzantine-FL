param(
    [string]$Dataset = "nyc-taxi"
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
$env:DATASET = $Dataset

Write-Host ""
Write-Host "============================================================"
Write-Host " Byzantine-Resilient FL - Full Experiment Suite"
Write-Host " Dataset : $Dataset"
Write-Host " Start   : $(Get-Date)"
Write-Host "============================================================"

Write-Host ""
Write-Host "Installing Python dependencies..."
try {
    python -m pip install numpy pandas scikit-learn scipy pyarrow fastparquet --quiet
} catch {
    Write-Host "Warning: pip install failed. Continuing..."
}

$resultsDir = Join-Path "results" $Dataset
$logDir = Join-Path $resultsDir "logs"

New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host ""
Write-Host "Checking dataset configuration..."

python -c "from dataset_config import DATASET_NAME,results_dir,tag; print('Dataset name:',DATASET_NAME); print('Results dir :',results_dir()); print('Sample file :',tag('table_baselines.csv'))"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Dataset configuration failed."
    exit 1
}

function Run-Experiment {
    param(
        [string]$Title,
        [string]$Script,
        [string]$LogFile
    )

    Write-Host ""
    Write-Host $Title

    & python $Script --dataset $Dataset 2>&1 | Tee-Object -FilePath $LogFile

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "$Script failed."
        exit $LASTEXITCODE
    }
}

Run-Experiment -Title "[1/5] Baseline comparison..." `
               -Script "experiment1_baselines.py" `
               -LogFile (Join-Path $logDir "${Dataset}_exp1.log")

Run-Experiment -Title "[2/5] Sensitivity analysis..." `
               -Script "experiment2_sensitivity.py" `
               -LogFile (Join-Path $logDir "${Dataset}_exp2.log")

Run-Experiment -Title "[3/5] Per-layer confusion breakdown..." `
               -Script "experiment3_layer_analysis.py" `
               -LogFile (Join-Path $logDir "${Dataset}_exp3.log")

Run-Experiment -Title "[4/5] Computational overhead..." `
               -Script "experiment4_overhead.py" `
               -LogFile (Join-Path $logDir "${Dataset}_exp4.log")

Run-Experiment -Title "[5/5] Adaptive adversary..." `
               -Script "experiment5_adaptive.py" `
               -LogFile (Join-Path $logDir "${Dataset}_exp5.log")

Write-Host ""
Write-Host "============================================================"
Write-Host " Complete : $(Get-Date)"
Write-Host " Dataset  : $Dataset"
Write-Host "============================================================"

Write-Host ""
Write-Host "Result files:"
Get-ChildItem -Path $resultsDir -Filter *.csv -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  $($_.Name)"
}

Write-Host ""
Write-Host "Done."
