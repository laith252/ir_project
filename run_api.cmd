@echo off
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "PYTHONPATH=.codex_deps;src"
set "IR_DATASETS_HOME=%CD%\data\raw\ir_datasets"
"%PYTHON_EXE%" -m uvicorn api:app --host 127.0.0.1 --port 8000
