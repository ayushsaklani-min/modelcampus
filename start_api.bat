@echo off
echo Starting API Server...
echo.
cd /d "%~dp0"
cd 12
python api_with_auth.py
pause

