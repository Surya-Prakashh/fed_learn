"""
fl_network_server.py — Global Federated Learning Server for Multi-Computer Training
===================================================================================
Coordinates 2 physical machines (Computer 1 & Computer 2):
  - Broadcasts global model parameters to participating computers.
  - Receives local parameter updates/deltas.
  - Aggregates using Federated Averaging (FedAvg).
  - Evaluates generalization on the Holdout test cohort (10 unseen WISDM subjects).
  - Streams real-time telemetry to the React Three Fiber 3D Visualization dashboard.
"""

import os
import io
import json
import asyncio
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARNet Architecture
# ─────────────────────────────────────────────────────────────────────────────
class HARNet(nn.Module):
    def __init__(self, input_dim=71, hidden_dim1=128, hidden_dim2=64, num_classes=5):
        super(HARNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Server Coordination State
# ─────────────────────────────────────────────────────────────────────────────
class FLServerManager:
    def __init__(self, total_rounds=20, clients_required=2):
        self.total_rounds = total_rounds
        self.clients_required = clients_required
        self.current_round = 1
        self.is_completed = False
        self.phase = "waiting_for_clients"  # 'broadcast', 'training', 'upload', 'aggregate', 'finished'

        self.input_dim = 71
        self.num_classes = 5
        self.global_model = HARNet(self.input_dim, num_classes=self.num_classes)

        # Holdout validation loader
        self.holdout_path = os.path.join("analysis_output", "holdout_test.pt")
        self.val_loader = None
        self.load_holdout_data()

        # Updates storage for the active round: client_id -> dict
        self.round_updates: Dict[str, dict] = {}
        self.history: List[dict] = []
        self.active_websockets: List[WebSocket] = []
        self.lock = asyncio.Lock()

        # Cumulative network transfer tracking
        self.cumulative_bytes = 0

    def load_holdout_data(self):
        if os.path.exists(self.holdout_path):
            data = torch.load(self.holdout_path, weights_only=False)
            ds = TensorDataset(data['X'], data['y'])
            self.val_loader = DataLoader(ds, batch_size=128, shuffle=False)
            self.input_dim = data.get('num_features', 71)
            self.num_classes = data.get('num_classes', 5)
            self.global_model = HARNet(self.input_dim, num_classes=self.num_classes)
            print(f"[Server] Loaded holdout test dataset with {len(ds)} samples.")
        else:
            print(f"[Server Warning] {self.holdout_path} not found. Run partition_data.py first.")

    def evaluate_global_model(self):
        if self.val_loader is None:
            return 0.0, 0.0
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in self.val_loader:
                out = self.global_model(X_b)
                loss = criterion(out, y_b)
                total_loss += loss.item() * len(y_b)
                _, pred = torch.max(out, 1)
                correct += (pred == y_b).sum().item()
                total += len(y_b)
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return float(avg_loss), float(acc)

    async def broadcast_ws(self, message: dict):
        dead_ws = []
        for ws in self.active_websockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead_ws.append(ws)
        for d in dead_ws:
            if d in self.active_websockets:
                self.active_websockets.remove(d)

    async def aggregate_round(self):
        self.phase = "aggregate"
        print(f"\n[Server] === Aggregating Round {self.current_round}/{self.total_rounds} ===")
        total_samples = sum(u['n_samples'] for u in self.round_updates.values())
        print(f"[Server] Received updates from {len(self.round_updates)} clients. Total samples: {total_samples}")

        # Weighted parameter aggregation
        new_state = {}
        for key in self.global_model.state_dict().keys():
            weighted_tensor = None
            for client_id, u in self.round_updates.items():
                weight = u['n_samples'] / total_samples
                client_param = u['state_dict'][key].float()
                if weighted_tensor is None:
                    weighted_tensor = weight * client_param
                else:
                    weighted_tensor += weight * client_param
            new_state[key] = weighted_tensor

        self.global_model.load_state_dict(new_state)

        # Evaluate global model on unseen subjects
        val_loss, val_acc = self.evaluate_global_model()

        # Communication volume (2 machines * model parameters size)
        param_bytes = sum(p.numel() * 4 for p in self.global_model.parameters())
        round_bytes = 2 * param_bytes * len(self.round_updates)
        self.cumulative_bytes += round_bytes

        client_metrics = {
            f"client_{cid}": {
                "loss": u.get("local_loss", 0.0),
                "acc": u.get("local_acc", 0.0),
                "samples": u.get("n_samples", 0)
            }
            for cid, u in self.round_updates.items()
        }

        round_record = {
            "round": self.current_round,
            "phase": "aggregate",
            "accuracy": round(val_acc, 4),
            "loss": round(val_loss, 4),
            "holdoutAccuracy": round(val_acc, 4),
            "holdoutLoss": round(val_loss, 4),
            "activeClients": [int(k) if k.isdigit() else k for k in self.round_updates.keys()],
            "clientMetrics": client_metrics,
            "bytes": round_bytes,
            "cumulativeBytes": self.cumulative_bytes,
            "totalRounds": self.total_rounds
        }
        self.history.append(round_record)
        print(f"[Server] Round {self.current_round} Summary: Holdout Acc = {val_acc*100:.2f}%, Loss = {val_loss:.4f}")

        # Send telemetry to WebSockets (React Three Fiber dashboard)
        await self.broadcast_ws(round_record)

        # Clear updates for next round
        self.round_updates.clear()

        if self.current_round >= self.total_rounds:
            self.is_completed = True
            self.phase = "finished"
            print(f"[Server] Training Complete! Final Generalized Accuracy: {val_acc*100:.2f}%")
            await self.broadcast_ws({"phase": "finished", "finalAccuracy": val_acc})
        else:
            self.current_round += 1
            self.phase = "broadcast"
            await self.broadcast_ws({
                "round": self.current_round,
                "phase": "broadcast",
                "activeClients": [1, 2]
            })


server_manager = FLServerManager(total_rounds=20, clients_required=2)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FastAPI App & Endpoints
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="WISDM Two-Computer Federated Learning Global Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METADATA_PATH = os.path.join("analysis_output", "client_metadata.json")

@app.get("/")
def read_root():
    return {
        "service": "WISDM Federated Learning Global Server",
        "current_round": server_manager.current_round,
        "total_rounds": server_manager.total_rounds,
        "clients_required": server_manager.clients_required,
        "phase": server_manager.phase,
        "is_completed": server_manager.is_completed
    }

@app.get("/api/fl/status")
def get_status():
    return {
        "current_round": server_manager.current_round,
        "total_rounds": server_manager.total_rounds,
        "phase": server_manager.phase,
        "waiting_for": list(server_manager.round_updates.keys()),
        "clients_required": server_manager.clients_required,
        "is_completed": server_manager.is_completed,
        "latest_history": server_manager.history[-1] if server_manager.history else None
    }

@app.get("/api/fl/model")
def get_global_model():
    """Returns the serialized PyTorch state_dict for the current round."""
    buf = io.BytesIO()
    torch.save(server_manager.global_model.state_dict(), buf)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={
            "X-Round": str(server_manager.current_round),
            "X-Total-Rounds": str(server_manager.total_rounds),
            "X-Phase": server_manager.phase,
            "X-Is-Completed": str(server_manager.is_completed)
        }
    )

@app.post("/api/fl/upload_update")
async def upload_update(
    client_id: str = Form(...),
    round_num: int = Form(...),
    n_samples: int = Form(...),
    local_loss: float = Form(0.0),
    local_acc: float = Form(0.0),
    file: UploadFile = File(...)
):
    """Receives model updates from Computer 1 or Computer 2."""
    if server_manager.is_completed:
        return JSONResponse({"status": "training_completed"}, status_code=200)

    if round_num != server_manager.current_round:
        return JSONResponse(
            {"status": "stale_round", "server_round": server_manager.current_round},
            status_code=400
        )

    content = await file.read()
    buf = io.BytesIO(content)
    client_state_dict = torch.load(buf, weights_only=False, map_location=torch.device('cpu'))

    async with server_manager.lock:
        server_manager.round_updates[str(client_id)] = {
            'state_dict': client_state_dict,
            'n_samples': n_samples,
            'local_loss': local_loss,
            'local_acc': local_acc
        }
        print(f"[Server] Received update from Client {client_id} for Round {round_num} (samples={n_samples}, acc={local_acc*100:.1f}%)")

        # Telemetry: notify UI that client uploaded
        await server_manager.broadcast_ws({
            "round": server_manager.current_round,
            "phase": "upload",
            "uploadingClient": client_id,
            "receivedCount": len(server_manager.round_updates),
            "clientsRequired": server_manager.clients_required
        })

        # When all required clients have submitted their updates, run aggregation
        if len(server_manager.round_updates) >= server_manager.clients_required:
            await server_manager.aggregate_round()

    return {"status": "accepted", "round": server_manager.current_round}

@app.get("/api/fl/partition/{client_id}")
def download_partition(client_id: int):
    """Allows Computer 2 to easily download its dataset partition directly from Computer 1."""
    part_path = os.path.join("analysis_output", f"client_{client_id}_data.pt")
    if not os.path.exists(part_path):
        raise HTTPException(status_code=404, detail=f"Partition file {part_path} not found. Run partition_data.py first.")
    return FileResponse(part_path, media_type="application/octet-stream", filename=f"client_{client_id}_data.pt")

@app.get("/api/client-metadata")
def get_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

@app.get("/api/metrics")
def get_metrics():
    """Provides historical benchmark data to the visualizer."""
    return {
        "federated": {
            "fedavg": server_manager.history
        }
    }

@app.websocket("/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    server_manager.active_websockets.append(websocket)
    print(f"[WebSocket] Connected client from {websocket.client}")
    try:
        # Send current round state upon connection
        if server_manager.history:
            for item in server_manager.history:
                await websocket.send_json(item)
        else:
            await websocket.send_json({
                "round": server_manager.current_round,
                "phase": server_manager.phase,
                "activeClients": [1, 2],
                "accuracy": 0.0,
                "loss": 2.0
            })
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in server_manager.active_websockets:
            server_manager.active_websockets.remove(websocket)
        print(f"[WebSocket] Client disconnected")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (0.0.0.0 listens to all LAN interfaces)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--rounds", type=int, default=20, help="Total FL rounds")
    args = parser.parse_args()

    server_manager.total_rounds = args.rounds
    print(f"============================================================")
    print(f"  Federated Learning Global Server Starting")
    print(f"  Listening on: http://{args.host}:{args.port}")
    print(f"  Total Rounds: {args.rounds} | Required Clients: {server_manager.clients_required}")
    print(f"============================================================")
    uvicorn.run(app, host=args.host, port=args.port)
