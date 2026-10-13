@echo off
setlocal enabledelayedexpansion
title Federated Learning - Computer 1 (Server + Client 1)

echo ============================================================
echo   FEDERATED LEARNING: COMPUTER 1 SETUP
echo ============================================================
echo.

:: Detect local IPv4 address
for /f "tokens=4" %%a in ('route print ^| findstr 0.0.0.0.*0.0.0.0') do (
    set LOCAL_IP=%%a
    goto :ip_found
)
:ip_found

echo [INFO] Computer 1 Local IP Address: !LOCAL_IP!
echo [INFO] On Computer 2, you will connect to: http://!LOCAL_IP!:8000
echo.

:: Check data partitions
if not exist "analysis_output\client_1_data.pt" (
    echo [INFO] Data partitions not found. Generating now...
    python partition_data.py
    if errorlevel 1 (
        echo [ERROR] Failed to partition data.
        pause
        exit /b 1
    )
)

echo [1/3] Starting Federated Global Server on port 8000...
start "FL Global Server (:8000)" cmd /k "python fl_network_server.py --host 0.0.0.0 --port 8000 --rounds 15"

:: Give server 2 seconds to initialize
timeout /t 2 /nobreak >nul

echo [2/3] Starting Local Edge Client 1...
echo Computer 1 will train Client 1 partition on-device and send to Global Server.
echo Waiting for Computer 2 to connect and train simultaneously...
echo.

python fl_network_client.py --server http://localhost:8000 --client-id 1 --epochs 2 --lr 0.01

echo.
echo ============================================================
echo   Training Session Ended
echo ============================================================
pause
