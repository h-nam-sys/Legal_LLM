@echo off
setlocal
cd /d "%~dp0.."
if not exist "env\Scripts\python.exe" (
    echo Project Python not found: env\Scripts\python.exe
    pause
    exit /b 1
)
"env\Scripts\python.exe" "testing\run_cacheless_tests.py" %*
set "qa_exit=%errorlevel%"
pause
exit /b %qa_exit%
