Set-Location -LiteralPath $PSScriptRoot
$python = "python"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
}
$env:PYTHONPATH = ".codex_deps;src"
$env:IR_DATASETS_HOME = Join-Path $PSScriptRoot "data\raw\ir_datasets"
& $python -m streamlit run app.py --global.developmentMode false --server.port 8501 --server.headless true
