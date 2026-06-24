Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot\.codex_deps;$PSScriptRoot\src"
$env:IR_DATASETS_HOME = "$PSScriptRoot\data\raw\ir_datasets"
$env:TMP = "$PSScriptRoot\.tmp"
$env:TEMP = "$PSScriptRoot\.tmp"

New-Item -ItemType Directory -Force -Path "$PSScriptRoot\.tmp" | Out-Null

$url = "http://127.0.0.1:8501"
$browserJob = Start-Job -ScriptBlock {
    param($TargetUrl)
    for ($attempt = 0; $attempt -lt 90; $attempt++) {
        try {
            Invoke-WebRequest -Uri $TargetUrl -UseBasicParsing -TimeoutSec 1 | Out-Null
            Start-Process $TargetUrl
            return
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }
} -ArgumentList $url

try {
    python -m streamlit run app.py `
        --global.developmentMode false `
        --server.address 127.0.0.1 `
        --server.port 8501 `
        --server.headless true `
        --browser.gatherUsageStats false
}
finally {
    Stop-Job -Job $browserJob -ErrorAction SilentlyContinue
    Remove-Job -Job $browserJob -Force -ErrorAction SilentlyContinue
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nProject failed to start. Keep this window open and review the error above." -ForegroundColor Red
    Read-Host "Press Enter to close"
}
