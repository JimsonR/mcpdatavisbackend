@echo off
REM Change to the project directory
cd "C:\New folder (5)\MCPSERVERLangchain"

call .venv\Scripts\activate.bat || exit /b
REM Run the Python script
python mathserver.py