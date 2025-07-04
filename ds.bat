echo off
REM Change to the project directory
cd /d "C:\New folder (5)\mcp-server-data-exploration" || exit /b
REM Activate the virtual environment
python src/mcp_server_ds/fastmcp_server.py || exit /b 