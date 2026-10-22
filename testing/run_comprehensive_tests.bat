@echo off
setlocal
cd /d "%~dp0.."
set "PYTHONIOENCODING=utf-8"
if exist "env\Scripts\python.exe" (
    "env\Scripts\python.exe" -u "%~dp0run_comprehensive_tests.py" %*
) else (
    python -u "%~dp0run_comprehensive_tests.py" %*
)
set "TEST_EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %TEST_EXIT_CODE%
