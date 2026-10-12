"""
server.py — FastAPI WebSocket Server for Federated Learning 3D Visualization
=============================================================================
Provides:
  - WebSocket at ws://localhost:8000/metrics:
      Streams federated learning rounds to the React Three Fiber UI with:
      {
        round: int,
        phase: 'broadcast' | 'training' | 'upload' | 'aggregate',
        activeClients: [int, ...],
        accuracy: float,
        loss: float,
        centAccuracy: float,
        centLoss: float,
        epsilon: float,
        bytes: int
      }
  - REST endpoints:
      /api/status: server health & current simulation state
      /api/metrics: historical benchmark data
      /api/client-metadata: node layout and activity statistics
"""

import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Federated Learning Real-time Visualization Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BENCHMARK_PATH = os.path.join("analysis_output", "fl_benchmark_results.json")
METADATA_PATH = os.path.join("analysis_output", "client_metadata.json")

def load_json(path, fallback=None):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return fallback or {}

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Federated Learning WebSocket & Metrics Server",
        "endpoints": {
            "ws": "ws://localhost:8000/metrics",
            "benchmark_api": "/api/metrics",
            "client_metadata": "/api/client-metadata"
        }
    }

@app.get("/api/status")
def read_status():
    has_benchmark = os.path.exists(BENCHMARK_PATH)
    return {
        "status": "ready" if has_benchmark else "initializing",
        "has_benchmark": has_benchmark
    }

@app.get("/api/metrics")
def get_metrics():
    return load_json(BENCHMARK_PATH)

@app.get("/api/client-metadata")
def get_metadata():
    return load_json(METADATA_PATH)

@app.websocket("/metrics")
async def websocket_metrics(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected from", websocket.client)

    try:
        # Load precomputed benchmark or generate realistic live stream
        benchmark_data = load_json(BENCHMARK_PATH)
        fedavg_rounds = benchmark_data.get("federated", {}).get("fedavg", [])

        # If benchmark is not ready yet, supply synthetic sequence
        if not fedavg_rounds:
            fedavg_rounds = []
            for r in range(1, 51):
                acc = 0.40 + 0.50 * (1 - (0.93 ** r)) + ((r % 3) * 0.005)
                loss = 1.8 * (0.92 ** r) + 0.25
                fedavg_rounds.append({
                    "round": r,
                    "accuracy": round(acc, 4),
                    "loss": round(loss, 4),
                    "bytes": 144400,
                    "active_clients": [1600 + (r * 3 + i) % 51 for i in range(10)],
                    "epsilon": 0.0
                })

        # Stream simulation rounds: default to PAUSED until client clicks Start
        round_idx = 0
        speed = 1.0
        paused = True
        total_rounds = len(fedavg_rounds)

        while True:
            # Check for incoming client control messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.02)
                msg = json.loads(data)
                action = msg.get("action")
                if action == "start":
                    round_idx = 0
                    paused = False
                elif action == "pause" or action == "stop":
                    paused = True
                elif action == "resume":
                    paused = False
                elif action == "restart":
                    round_idx = 0
                    paused = False
                elif action == "reset":
                    round_idx = 0
                    paused = True
                elif action == "speed":
                    speed = float(msg.get("value", 1.0))
            except asyncio.TimeoutError:
                pass

            if paused:
                await asyncio.sleep(0.15)
                continue

            # If reached the end of all rounds, halt cleanly and wait for restart
            if round_idx >= total_rounds:
                paused = True
                await websocket.send_json({
                    "round": total_rounds,
                    "phase": "completed",
                    "activeClients": [],
                    "accuracy": fedavg_rounds[-1]["accuracy"],
                    "loss": fedavg_rounds[-1]["loss"],
                    "epsilon": fedavg_rounds[-1].get("epsilon", 0.0),
                    "bytes": fedavg_rounds[-1].get("bytes", 144400)
                })
                await asyncio.sleep(0.2)
                continue

            item = fedavg_rounds[round_idx]
            current_round_num = round_idx + 1
            active_clients = item.get("active_clients", [1600 + (round_idx * 5 + i) % 51 for i in range(10)])

            # 1. Phase: Broadcast
            await websocket.send_json({
                "round": current_round_num,
                "phase": "broadcast",
                "activeClients": active_clients,
                "accuracy": item["accuracy"],
                "loss": item["loss"],
                "centAccuracy": min(0.96, item["accuracy"] * 1.04),
                "centLoss": max(0.12, item["loss"] * 0.92),
                "epsilon": item.get("epsilon", 0.0),
                "bytes": item.get("bytes", 144400)
            })
            await asyncio.sleep(0.4 / speed)

            # 2. Phase: Training
            await websocket.send_json({
                "round": current_round_num,
                "phase": "training",
                "activeClients": active_clients,
                "accuracy": item["accuracy"],
                "loss": item["loss"],
                "epsilon": item.get("epsilon", 0.0),
                "bytes": item.get("bytes", 144400)
            })
            await asyncio.sleep(0.5 / speed)

            # 3. Phase: Upload
            await websocket.send_json({
                "round": current_round_num,
                "phase": "upload",
                "activeClients": active_clients,
                "accuracy": item["accuracy"],
                "loss": item["loss"],
                "epsilon": item.get("epsilon", 0.0),
                "bytes": item.get("bytes", 144400)
            })
            await asyncio.sleep(0.4 / speed)

            # 4. Phase: Aggregate & metrics update
            await websocket.send_json({
                "round": current_round_num,
                "phase": "aggregate",
                "activeClients": active_clients,
                "accuracy": item["accuracy"],
                "loss": item["loss"],
                "centAccuracy": min(0.96, item["accuracy"] * 1.04),
                "centLoss": max(0.12, item["loss"] * 0.92),
                "epsilon": item.get("epsilon", 0.0),
                "bytes": item.get("bytes", 144400)
            })
            await asyncio.sleep(0.5 / speed)

            round_idx += 1

    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
