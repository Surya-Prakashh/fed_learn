# Federated Learning for Wearable Human Activity Recognition (WISDM)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-r186-black.svg?logo=three.js&logoColor=white)](https://threejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, privacy-preserving **Federated Learning (FL)** platform and **3D interactive visualizer** trained on the **WISDM Human Activity Recognition dataset** (51 subjects, 15.6+ million sensor readings).

---

## 📌 Executive Summary

Wearable sensors on smartphones and smartwatches capture continuous accelerometer and gyroscope biometrics that reveal private daily routines, gait speed, and health symptoms. Centralized model training requires aggregating raw sensor streams onto a cloud server, introducing severe data custody and privacy risks.

This repository implements a decentralized Federated Learning solution where **raw sensor data never leaves edge devices**:
- **51 Edge Clients**: Natural client partitions representing the 51 human subjects in WISDM with real hardware heterogeneity (pocket smartphones vs. wrist smartwatches).
- **Core Algorithms**: Centralized baseline, Local-Only baseline, **FedAvg**, and **FedProx** for statistical non-IID client drift mitigation.
- **Privacy Suite**: $(\epsilon, \delta)$-Differential Privacy with noise calibration, pairwise zero-sum **Secure Aggregation**, and **Membership Inference Attack (MIA)** evaluation.
- **Bandwidth Compression**: FP32 vs. 8-bit Integer Quantization (INT8, $4\times$ savings) vs. Top-10% gradient sparsification ($6.7\times$ savings).
- **Interactive 3D Web Dashboard**: Built with React 19, React Three Fiber, and Recharts, featuring live WebSocket telemetry and interactive model inference.

---

## 🎯 What We Are Predicting

The system performs real-time **Human Activity Recognition** across 5 physiological activity groups:

| Activity Group | Included Sub-Activities | Sensor Profile |
| :--- | :--- | :--- |
| 🏃 **Locomotion** | Walking, Jogging, Climbing Stairs | High cyclic vertical acceleration ($1.2–3.2\text{ Hz}$ cadence) |
| 🪑 **Sedentary** | Sitting, Standing Still | Static $1.0\text{G}$ gravity alignment, near-zero variance |
| ⌨️ **Hand-Oriented** | Typing, Writing, Brushing Teeth, Clapping, Folding | Low torso displacement, continuous wrist gyroscope rotations |
| 🍽️ **Eating** | Eating Soup, Pasta, Chips, Sandwich, Drinking | Periodic hand-to-mouth elevation angles (~$0.8\text{ Hz}$) |
| ⚽ **Sports** | Kicking Soccer Ball, Catching, Dribbling Basketball | High kinetic energy bursts, rapid multi-axial acceleration spikes |

Signals are segmented into **10-second windows** (50% overlap), extracting **71 kinematic, statistical, and spectral (FFT) features** per window.

---

## 📊 Benchmark & Empirical Performance

All models were evaluated on an unseen holdout cohort (10 unseen subjects) to measure genuine cross-subject generalization:

| Strategy / Setup | Training Mode | Local Acc. | Holdout Generalization | Holdout Loss | Total Network Traffic (30 Rds) | Privacy Guarantee |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | Pooled Cloud Training | 82.39% | **57.05%** | 1.550 | Centralized | None (High Leakage) |
| **Local-Only Baseline** | Disconnected (Each Client Alone) | **91.50%** | **36.67%** (Failure) | 2.140 | 0 MB | Private |
| **FedAvg (FP32)** | Distributed ($K=8$ / rd) | 85.12% | **50.48%** | 1.363 | 33.29 MB | Gradient only |
| **FedProx ($\mu=0.01$)** | Proximal Regularizer | 86.40% | **56.34%** | 1.346 | 33.29 MB | Gradient + Anti-Drift |
| **FedAvg-INT8** | 8-Bit Quantized | 84.80% | **56.01%** | 1.189 | **8.32 MB ($4\times$ reduction)** | Compressed |
| **FedAvg-TopK10** | 90% Gradient Sparsity | 85.05% | **56.69%** | 1.245 | **4.99 MB ($6.7\times$ reduction)** | Compressed |
| **DP-FedAvg ($\epsilon=10$)** | Mild Gaussian Noise | 78.10% | **44.48%** | 1.286 | 33.29 MB | $(10, 10^{-4})$-DP |
| **DP-FedAvg ($\epsilon=1$)** | Strict Cryptographic DP | 55.20% | **28.40%** | 189.34 | 33.29 MB | $(1, 10^{-4})$-DP (Immune to MIA) |

### Key Findings:
1. **The Local-Only Generalization Trap**: Individual devices training strictly on their own data overfit heavily to user gait and habits ($91.5\%$ local accuracy), but collapse to **$36.67\%$** on unseen subjects.
2. **FedProx Rescues Non-IID Client Drift**: On heterogeneous client partitions, standard FedAvg plateaus at $50.48\%$. Adding the proximal penalty $\frac{\mu}{2} ||w - w_t||^2$ pulls updates toward global consensus, restoring accuracy to **$56.34\%$** (matching the centralized ceiling).
3. **Defense Against Membership Inference**: A shadow loss attacker against the Centralized model achieves an AUC of **$0.578$** ($60.47\%$ accuracy). Under DP-FedAvg ($\epsilon=1$), attacker AUC drops to **$0.036$** ($53.50\%$ accuracy, near random guess).
4. **Bandwidth Optimization**: Top-10% sparsification drops cumulative bandwidth from $33.29\text{ MB}$ to **$4.99\text{ MB}$** with zero loss in accuracy ($56.69\%$).

---

## 💻 Interactive 3D Web Application (`federated-viz`)

The web frontend offers three tabs:

### 1. 🌐 3D Live Network
- **51 Client Nodes**: Real WISDM subjects orbiting the central server, color-coded by dominant activity with hover telemetry.
- **Round Animations**: Broadcast wave $\to$ On-device pulse training $\to$ Particle beam upload $\to$ Server aggregation flash.
- **Progress Bar**: Displays session progress (`Round X / 50 • Y%`) and active phase, automatically halting on target convergence.
- **Real-Time Recharts**: Live accuracy area charts, training loss, and cumulative network transfer (KB).

### 2. 🔮 Live Inference & Activity Prediction
- Test the trained global model on custom sensor inputs or 6 real-world presets (*Jogging, Normal Walking, Typing, Sitting, Eating, Sports*).
- Real-time softmax probability bars across all 5 classes with kinematic explanations.

### 3. 📊 Benchmark & Dataset Analysis
- Complete empirical tables, KPIs, and publication-ready figures for client distributions, device heterogeneity, DP sweeps, and communication compression.

---

## 🚀 Quickstart

### Prerequisites
- **Python**: Version 3.10 or higher
- **Node.js**: Version 18 or higher (with `npm`)

### 1. One-Click Launch (Windows)
Double-click [`run.bat`](file:///d:/Education/DPSA_proj/run.bat) or run in terminal:
```cmd
.\run.bat
```
This script will:
1. Verify Python and Node.js environments.
2. Automatically clean up stale port bindings.
3. Start the Python WebSocket server on **port 8000**.
4. Start the Vite React Three Fiber frontend on **port 5173**.
5. Launch your browser at `http://localhost:5173/`.

To shut down services:
```cmd
.\stop.bat
```

### 2. Manual Startup

**Backend Server**:
```bash
python -m pip install fastapi uvicorn websockets torch numpy pandas matplotlib seaborn scikit-learn
python server.py
```

**Frontend Dashboard**:
```bash
cd federated-viz
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 3. Re-Running the PyTorch ML Benchmark
To regenerate all experimental figures and metrics from scratch:
```bash
python fl_pipeline.py
```
Outputs are saved directly to `analysis_output/` and copied to `federated-viz/public/`.

---

## 📁 Repository Structure

```
├── data_analysis.py            # WISDM feature extraction & distribution analyzer
├── fl_pipeline.py              # PyTorch FL pipeline (FedAvg, FedProx, DP, SecAgg, MIA, Compression)
├── server.py                   # FastAPI WebSocket server (ws://localhost:8000/metrics)
├── run.bat                     # Windows one-click launcher
├── stop.bat                    # One-click process terminator
├── run_pipeline.bat            # One-click benchmark pipeline runner
├── federated_learning_wisdm_report.md # Comprehensive technical report
├── analysis_output/            # Exported figures, metadata, and benchmark JSONs
│   ├── fl_benchmark_results.json
│   ├── client_metadata.json
│   ├── fl_vs_centralized_vs_local.png
│   ├── differential_privacy_sweep.png
│   ├── communication_compression.png
│   ├── membership_inference_attack_roc.png
│   └── secure_aggregation_demo.png
└── federated-viz/              # React Three Fiber 3D Dashboard
    ├── src/
    │   ├── App.jsx             # Main dashboard with 3 tabs & progress tracker
    │   ├── FederatedScene.jsx  # Three.js 3D scene (51 client nodes & particles)
    │   ├── PredictTab.jsx      # Interactive live model inference & presets
    │   ├── AnalysisDashboard.jsx # Comprehensive empirical analysis view
    │   ├── MetricsCharts.jsx   # Live Recharts visualization
    │   └── simulation.js       # In-browser neural network FedAvg engine
    └── package.json
```

---

## 📜 License
This project is licensed under the MIT License.
