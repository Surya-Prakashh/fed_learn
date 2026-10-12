@echo off
title Run Federated Learning PyTorch Benchmark Pipeline
echo ======================================================================
echo    RUNNING WISDM FEDERATED LEARNING PYTORCH BENCHMARK PIPELINE
echo ======================================================================
echo.

set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%"

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH!
    pause
    exit /b 1
)

echo Starting benchmark pipeline (Centralized, Local-Only, FedAvg, FedProx, DP, Secure Agg, MIA, Quantization)...
python -u fl_pipeline.py

echo.
echo Benchmark run finished! Output plots and JSON are in analysis_output/
pause
