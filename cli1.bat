@echo off
REM Change to the project directory
cd /d "C:\New folder (5)\MCPSERVERLangchain" || exit /b

REM Activate the virtual environment
call .venv\Scripts\activate.bat || exit /b

REM Run the Python script
python cli1.py
