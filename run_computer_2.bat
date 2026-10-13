@echo off
setlocal enabledelayedexpansion
title Federated Learning - Computer 2 (Edge Client 2)

echo ============================================================
echo   FEDERATED LEARNING: COMPUTER 2 (EDGE CLIENT)
echo ============================================================
echo.

set SERVER_IP=%1

if "%SERVER_IP%"=="" (
    set /p SERVER_IP="Enter Computer 1 IP Address (e.g. 192.168.1.5): "
)

if "%SERVER_IP%"=="" (
    echo [ERROR] Server IP cannot be empty.
    pause
    exit /b 1
)

set SERVER_URL=http://%SERVER_IP%:8000

echo.
echo [INFO] Connecting to Global Server at: %SERVER_URL%
echo [INFO] Client 2 will automatically download its training partition from Computer 1.
echo.

python fl_network_client.py --server %SERVER_URL% --client-id 2 --epochs 2 --lr 0.01

echo.
echo ============================================================
echo   Client 2 Training Session Ended
echo ============================================================
pause
