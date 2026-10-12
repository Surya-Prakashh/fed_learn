@echo off
title Stop Federated Learning System
echo ======================================================================
echo    STOPPING FEDERATED LEARNING SERVERS (PORT 8000 AND 5173)
echo ======================================================================
echo.

echo Stopping processes listening on Port 8000 (Python WebSocket Server)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    echo Killing PID %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
)

echo Stopping processes listening on Port 5173 (Vite Dev Server)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING 2^>nul') do (
    echo Killing PID %%a on port 5173...
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo All Federated Learning services stopped successfully.
ping 127.0.0.1 -n 3 >nul
