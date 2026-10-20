@echo off
setlocal
set "PYTHON_CMD=%~dp0..\graspkit-tools\.venv\Scripts\python.exe"
if not exist "%PYTHON_CMD%" (
    echo Set up the shared environment from graspkit-tools with uv sync --extra cpu.
    exit /b 1
)
"%PYTHON_CMD%" "%~dp0build_package.py" %*
exit /b %ERRORLEVEL%
