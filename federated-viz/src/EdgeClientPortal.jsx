import React, { useState, useEffect } from 'react';

export default function EdgeClientPortal() {
  const [selectedClient, setSelectedClient] = useState('2');
  const [status, setStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [logs, setLogs] = useState([
    { time: '00:00:01', text: 'Edge Client Portal connected to Global Server.' }
  ]);
  const [localMetrics, setLocalMetrics] = useState({
    acc: 0.5694,
    loss: 1.0176,
    samples: 20246,
    subjects: 21
  });

  const apiHost = window.location.hostname || 'localhost';
  const apiUrl = `http://${apiHost}:8000`;

  const addLog = (text) => {
    const time = new Date().toTimeString().split(' ')[0];
    setLogs((prev) => [...prev.slice(-20), { time, text }]);
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/fl/status`);
      const data = await res.json();
      setStatus(data);
      if (data.clients && data.clients[selectedClient]) {
        const c = data.clients[selectedClient];
        if (c.last_acc) {
          setLocalMetrics({
            acc: c.last_acc,
            loss: c.last_loss || 1.0,
            samples: c.samples || (selectedClient === '2' ? 20246 : 18266),
            subjects: selectedClient === '2' ? 21 : 20
          });
        }
      }
    } catch (e) {
      console.warn("Client status fetch error", e);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${apiHost}:8000/metrics`);
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'new_round_started' || msg.phase === 'broadcast') {
          addLog(`★ Server broadcasted Round ${msg.round}! Local client ready for SGD.`);
          fetchStatus();
        } else if (msg.type === 'round_completed' || msg.phase === 'aggregate') {
          addLog(`Global Aggregation complete (Holdout Accuracy: ${(msg.accuracy * 100).toFixed(2)}%).`);
          fetchStatus();
        }
      } catch (e) {}
    };

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [selectedClient]);

  const handleTrainLocal = async () => {
    setIsTraining(true);
    addLog(`[Client ${selectedClient}] Pulling global parameters from ${apiUrl}...`);
    addLog(`[Client ${selectedClient}] Training on-device on ${localMetrics.samples.toLocaleString()} sensor windows...`);

    try {
      const res = await fetch(`${apiUrl}/api/fl/client_train/${selectedClient}`, { method: 'POST' });
      const data = await res.json();
      setLocalMetrics((prev) => ({
        ...prev,
        acc: data.acc,
        loss: data.loss
      }));
      addLog(`[Client ${selectedClient}] Training finished! Loss: ${data.loss.toFixed(4)}, Local Acc: ${(data.acc * 100).toFixed(2)}%`);
      addLog(`[Client ${selectedClient}] Sent updated model weights to Global Server.`);
      fetchStatus();
    } catch (e) {
      addLog(`[Client ${selectedClient}] Error during training: ${e}`);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div style={{
      position: 'absolute',
      inset: '72px 0 0 0',
      zIndex: 50,
      background: '#0a0e1a',
      overflowY: 'auto',
      padding: '24px 32px 64px'
    }}>
      <div style={{ maxWidth: '900px', margin: '0 auto', color: '#f3f4f6' }}>
        {/* Top Header Card */}
        <div style={{
        background: 'rgba(19, 25, 41, 0.85)',
        backdropFilter: 'blur(16px)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '16px',
        padding: '20px 28px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '44px', height: '44px', borderRadius: '12px',
            background: 'linear-gradient(135deg, #06b6d4, #3b82f6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 800, fontSize: '18px'
          }}>
            EDGE
          </div>
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700 }}>
              {selectedClient === '2' ? 'Computer 2 (Remote Client Portal)' : 'Computer 1 (Host Client Portal)'}
            </h2>
            <p style={{ fontSize: '12px', color: '#9ca3af' }}>
              Connected to Global Server at <span style={{ fontFamily: 'monospace', color: '#38bdf8' }}>{apiUrl}</span>
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span style={{ fontSize: '12px', color: '#9ca3af' }}>Switch Role:</span>
          <select
            value={selectedClient}
            onChange={(e) => setSelectedClient(e.target.value)}
            style={{
              background: 'rgba(255, 255, 255, 0.06)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: '#fff',
              padding: '8px 14px',
              borderRadius: '8px',
              fontSize: '13px',
              cursor: 'pointer'
            }}
          >
            <option value="2">Computer 2 (Remote Node)</option>
            <option value="1">Computer 1 (Host Node)</option>
          </select>
          <span style={{
            padding: '4px 10px',
            borderRadius: '20px',
            fontSize: '11px',
            fontWeight: 700,
            background: 'rgba(16, 185, 129, 0.15)',
            color: '#34d399',
            border: '1px solid rgba(16, 185, 129, 0.3)'
          }}>
            CONNECTED
          </span>
        </div>
      </div>

      {/* Dataset & Sensor Partition Card */}
      <div style={{
        background: 'rgba(19, 25, 41, 0.85)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '20px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 700 }}>📊 On-Device Partition & Sensor Profile</h3>
          <span style={{ fontSize: '13px', color: '#38bdf8', fontFamily: 'monospace' }}>
            {localMetrics.samples.toLocaleString()} Sensor Windows (10s @ 20Hz)
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '16px' }}>
          <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
            <div style={{ fontSize: '11px', color: '#9ca3af', textTransform: 'uppercase', fontWeight: 600 }}>Assigned Subjects</div>
            <div style={{ fontSize: '24px', fontWeight: 800, fontFamily: 'monospace', color: '#38bdf8', marginTop: '4px' }}>
              {localMetrics.subjects} Subjects
            </div>
            <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '2px' }}>WISDM Heterogeneous Cohort</div>
          </div>

          <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
            <div style={{ fontSize: '11px', color: '#9ca3af', textTransform: 'uppercase', fontWeight: 600 }}>Local Accuracy</div>
            <div style={{ fontSize: '24px', fontWeight: 800, fontFamily: 'monospace', color: '#34d399', marginTop: '4px' }}>
              {(localMetrics.acc * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '2px' }}>On-Device Test Set</div>
          </div>

          <div style={{ background: 'rgba(255, 255, 255, 0.03)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
            <div style={{ fontSize: '11px', color: '#9ca3af', textTransform: 'uppercase', fontWeight: 600 }}>Training Loss</div>
            <div style={{ fontSize: '24px', fontWeight: 800, fontFamily: 'monospace', color: '#fbbf24', marginTop: '4px' }}>
              {localMetrics.loss.toFixed(3)}
            </div>
            <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '2px' }}>Cross-Entropy Error</div>
          </div>
        </div>

        <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            background: 'linear-gradient(90deg, #06b6d4, #10b981)',
            width: isTraining ? '65%' : '100%',
            transition: 'width 0.3s'
          }} />
        </div>
      </div>

      {/* On-Device Training Action Card */}
      <div style={{
        background: 'rgba(19, 25, 41, 0.85)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '20px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h3 style={{ fontSize: '16px', fontWeight: 700 }}>⚡ On-Device Federated Training Station</h3>
          <span style={{ fontSize: '13px', color: '#34d399', fontWeight: 600 }}>
            Global Round: R {status?.current_round || 1} / {status?.total_rounds || 20}
          </span>
        </div>
        <p style={{ fontSize: '13px', color: '#9ca3af', marginBottom: '16px', lineHeight: 1.5 }}>
          Execute local gradient descent on this computer's private sensor data partition. Only parameter weights are transmitted to the Global Server; raw sensor biometrics never leave the device.
        </p>

        <button
          onClick={handleTrainLocal}
          disabled={isTraining}
          style={{
            width: '100%',
            padding: '16px',
            borderRadius: '12px',
            border: 'none',
            background: isTraining ? '#4b5563' : 'linear-gradient(135deg, #06b6d4, #2563eb)',
            color: '#fff',
            fontSize: '15px',
            fontWeight: 700,
            cursor: isTraining ? 'not-allowed' : 'pointer',
            boxShadow: isTraining ? 'none' : '0 4px 20px rgba(6, 182, 212, 0.3)',
            transition: 'all 0.2s'
          }}
        >
          {isTraining ? '⏳ Executing On-Device SGD & Uploading...' : '🚀 Train Model Locally & Send to Global Server'}
        </button>
      </div>

      {/* Device Activity Log */}
      <div style={{
        background: 'rgba(19, 25, 41, 0.85)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '16px',
        padding: '20px 24px'
      }}>
        <h4 style={{ fontSize: '14px', fontWeight: 700, marginBottom: '12px' }}>📜 On-Device Activity & Telemetry Log</h4>
        <div style={{
          background: 'rgba(0, 0, 0, 0.35)',
          borderRadius: '10px',
          padding: '12px',
          fontFamily: 'monospace',
          fontSize: '12px',
          height: '140px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px'
        }}>
          {logs.map((l, i) => (
            <div key={i} style={{ color: '#d1d5db' }}>
              <span style={{ color: '#6b7280', marginRight: '6px' }}>[{l.time}]</span> {l.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  </div>
  );
}
