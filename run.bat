@echo off
setlocal enabledelayedexpansion
title Federated Learning System Launcher

echo ======================================================================
echo    FEDERATED LEARNING 3D VISUALIZATION AND BENCHMARK SYSTEM
echo ======================================================================
echo.

set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%"

:: 1. Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: 2. Check Node / npm
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] npm / Node.js is not found in PATH!
    echo Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

:: 3. Automatically free port 8000 if previously occupied
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    echo [INFO] Port 8000 is currently occupied by PID %%a. Releasing port...
    taskkill /F /PID %%a >nul 2>&1
)

:: 4. Check frontend dependencies
if not exist "%ROOT_DIR%federated-viz\node_modules" (
    echo [INFO] Installing frontend dependencies in federated-viz...
    cd /d "%ROOT_DIR%federated-viz"
    call npm install
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install npm dependencies.
        pause
        exit /b 1
    )
    cd /d "%ROOT_DIR%"
)

echo [1/3] Starting Python WebSocket Server (Port 8000)...
start "FL WebSocket Server (Port 8000)" cmd /k "cd /d "%ROOT_DIR%" && title FL Server :8000 && python server.py"

:: Wait 2 seconds for backend to bind port (using ping for maximum compatibility)
ping 127.0.0.1 -n 3 >nul

echo [2/3] Starting React Three Fiber Vite App (Port 5173)...
start "FL 3D Visualizer (Port 5173)" cmd /k "cd /d "%ROOT_DIR%federated-viz" && title FL Frontend :5173 && npm run dev"

:: Wait 3 seconds for Vite server to boot
ping 127.0.0.1 -n 4 >nul

echo [3/3] Launching Web Browser at http://localhost:5173/...
start http://localhost:5173/

echo.
echo ======================================================================
echo    SERVICES STARTED SUCCESSFULLY!
echo ======================================================================
echo  - Backend / WebSocket : http://localhost:8000 (ws://localhost:8000/metrics)
echo  - 3D Visualizer       : http://localhost:5173
echo.
echo  To shut down both servers, close the opened terminal windows or run:
echo    stop.bat
echo ======================================================================
echo.
pause
