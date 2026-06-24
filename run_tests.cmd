@echo off
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "PYTHONPATH=.codex_deps;src"
set "IR_DATASETS_HOME=%CD%\data\raw\ir_datasets"
set "MPLCONFIGDIR=.tmp\matplotlib"
"%PYTHON_EXE%" -m unittest discover -s tests -v
