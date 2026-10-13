"""
fl_network_client.py — Federated Learning Edge Client for Computer 1 / Computer 2
==================================================================================
Runs on any computer participating in the Federated Learning training:
  1. Connects to the Global Server over LAN or localhost HTTP.
  2. Downloads its dataset partition automatically if not present locally.
  3. In each round:
     - Downloads latest global model weights.
     - Trains locally on its assigned WISDM sensor data partition.
     - Sends local model weights and sample counts back to the server.
     - Waits for the round aggregation to finish before advancing.
"""

import os
import io
import time
import argparse
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# 1. HARNet Architecture (Identical to Server)
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Local Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_local_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X_b, y_b in dataloader:
        optimizer.zero_grad()
        out = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_b)
        _, pred = torch.max(out, 1)
        correct += (pred == y_b).sum().item()
        total += len(y_b)

    return total_loss / max(1, total), correct / max(1, total)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main Client Execution
# ─────────────────────────────────────────────────────────────────────────────
def run_client(server_url: str, client_id: str, local_epochs: int = 2, lr: float = 0.01, batch_size: int = 64, data_path: str = None):
    server_url = server_url.rstrip("/")
    print(f"\n============================================================")
    print(f"  Federated Learning Edge Client (ID: {client_id})")
    print(f"  Target Server: {server_url}")
    print(f"  Local Epochs: {local_epochs} | LR: {lr} | Batch Size: {batch_size}")
    print(f"============================================================\n")

    # 1. Ensure local dataset is ready
    if not data_path:
        data_path = os.path.join("analysis_output", f"client_{client_id}_data.pt")

    if not os.path.exists(data_path):
        print(f"[Client {client_id}] Local partition '{data_path}' not found. Downloading from server...")
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        download_url = f"{server_url}/api/fl/partition/{client_id}"
        resp = requests.get(download_url, timeout=30)
        if resp.status_code == 200:
            with open(data_path, "wb") as f:
                f.write(resp.content)
            print(f"[Client {client_id}] Downloaded partition successfully to {data_path} ({len(resp.content)} bytes).")
        else:
            raise RuntimeError(f"Failed to fetch data partition from {download_url}: HTTP {resp.status_code}")

    print(f"[Client {client_id}] Loading training dataset from {data_path}...")
    local_data = torch.load(data_path, weights_only=False)
    X, y = local_data['X'], local_data['y']
    num_features = local_data.get('num_features', 71)
    num_classes = local_data.get('num_classes', 5)
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"[Client {client_id}] Dataset loaded: {len(X)} samples, {num_features} features, {num_classes} classes.")

    model = HARNet(input_dim=num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()

    last_completed_round = 0

    # 2. Main Federated Training Loop
    while True:
        try:
            # Check server status
            status_resp = requests.get(f"{server_url}/api/fl/status", timeout=10)
            if status_resp.status_code != 200:
                print(f"[Client {client_id}] Waiting for server to become ready...")
                time.sleep(2)
                continue

            status = status_resp.json()
            server_round = status["current_round"]
            is_completed = status["is_completed"]

            if is_completed:
                print(f"\n[Client {client_id}] Server announced training is COMPLETE! Exiting.")
                break

            # If we already trained on this round, wait for aggregation
            if server_round <= last_completed_round:
                time.sleep(1)
                continue

            print(f"\n[Client {client_id}] >>> Commencing Round {server_round}/{status['total_rounds']} <<<")

            # Download global model weights
            model_resp = requests.get(f"{server_url}/api/fl/model", timeout=20)
            if model_resp.status_code != 200:
                print(f"[Client {client_id}] Failed to fetch model. Retrying...")
                time.sleep(1)
                continue

            # Load global weights into local model
            state_buf = io.BytesIO(model_resp.content)
            global_state = torch.load(state_buf, weights_only=False, map_location=torch.device('cpu'))
            model.load_state_dict(global_state)

            # Local SGD training
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            avg_loss, avg_acc = 0.0, 0.0
            print(f"[Client {client_id}] Training on-device for {local_epochs} local epochs...")
            for ep in range(1, local_epochs + 1):
                avg_loss, avg_acc = train_local_epoch(model, train_loader, optimizer, criterion)
                print(f"  [Epoch {ep}/{local_epochs}] Loss: {avg_loss:.4f} | Local Acc: {avg_acc*100:.2f}%")

            # Serialize trained weights
            upload_buf = io.BytesIO()
            torch.save(model.state_dict(), upload_buf)
            upload_buf.seek(0)

            # Upload to server
            print(f"[Client {client_id}] Uploading model parameters to Global Server...")
            files = {'file': ('weights.pt', upload_buf, 'application/octet-stream')}
            data = {
                'client_id': str(client_id),
                'round_num': server_round,
                'n_samples': len(X),
                'local_loss': float(avg_loss),
                'local_acc': float(avg_acc)
            }
            upload_resp = requests.post(f"{server_url}/api/fl/upload_update", files=files, data=data, timeout=30)

            if upload_resp.status_code == 200:
                print(f"[Client {client_id}] Round {server_round} update successfully uploaded!")
                last_completed_round = server_round
            else:
                print(f"[Client {client_id}] Upload response: {upload_resp.text}")

            # Wait a moment before checking next round status
            time.sleep(1.5)

        except requests.exceptions.RequestException as e:
            print(f"[Client {client_id}] Network connection error: {e}. Retrying in 3 seconds...")
            time.sleep(3)
        except Exception as e:
            print(f"[Client {client_id}] Unexpected error: {e}. Retrying in 2 seconds...")
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Edge Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Global Server URL (e.g. http://192.168.1.5:8000)")
    parser.add_argument("--client-id", default="1", help="Client Identifier (e.g. 1 or 2)")
    parser.add_argument("--epochs", type=int, default=2, help="Local training epochs per round")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--data-path", default=None, help="Path to local data partition (.pt file)")
    args = parser.parse_args()

    run_client(
        server_url=args.server,
        client_id=args.client_id,
        local_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        data_path=args.data_path
    )
