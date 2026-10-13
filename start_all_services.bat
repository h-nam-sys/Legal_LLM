@echo off
setlocal
rem Run from the Legal_LLM project root.

if not exist "env\Scripts\activate.bat" (
    echo ERROR: env\Scripts\activate.bat was not found. Run this from the project root.
    pause
    exit /b 1
)
if not exist "llm_service\main.py" (
    echo ERROR: llm_service\main.py was not found.
    pause
    exit /b 1
)
if not exist "backend\main.py" (
    echo ERROR: backend\main.py was not found.
    pause
    exit /b 1
)
if not exist "backend\admin_dashboard.py" (
    echo ERROR: backend\admin_dashboard.py was not found.
    pause
    exit /b 1
)

call "env\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Could not activate the virtual environment.
    pause
    exit /b 1
)
set "PYTHONIOENCODING=utf-8"

rem Child windows inherit the activated virtual environment.
start "Legal LLM - LLM Service" cmd /d /k python llm_service/main.py
start "Legal LLM - Backend" /d "%CD%\backend" cmd /d /k python -m uvicorn main:app --port 8000 --reload
start "Legal LLM - Admin Dashboard" /d "%CD%\backend" cmd /d /k python -m streamlit run admin_dashboard.py --server.address 127.0.0.1 --server.port 8501

echo Opened three service windows. Model loading may take a while.
echo Admin dashboard: http://127.0.0.1:8501
echo Stop each service with Ctrl+C in its window.
endlocal
