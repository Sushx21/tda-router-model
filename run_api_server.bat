@echo off
cd /d "%~dp0"
set PORT=8000
if not "%~1"=="" set PORT=%~1

echo Opening Windows Firewall port %PORT% (if running as Administrator)...
netsh advfirewall firewall add rule name="TDA Router API Port %PORT%" dir=in action=allow protocol=TCP localport=%PORT% >nul 2>&1

echo Starting TDA Router API Server on 0.0.0.0:%PORT% ...
python api_server.py --host 0.0.0.0 --port %PORT%
pause
