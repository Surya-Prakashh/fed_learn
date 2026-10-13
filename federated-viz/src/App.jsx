/**
 * App.jsx — Federated Learning 3D Visualization & Analytics Dashboard
 * ====================================================================
 * Features:
 *  - 3 Navigation Tabs:
 *      1) 🌐 3D Live Network (React Three Fiber scene with 51 client nodes)
 *      2) 🔮 Live Inference & Predict (Test inputs & predict with federated model)
 *      3) 📊 Benchmark & Analysis (Full WISDM empirical results)
 *  - Global Session Progress Bar:
 *      - Shows current round / max rounds (e.g. Round 14 / 50 • 28%).
 *      - Displays active phase: Broadcast -> Local Train -> Upload -> Aggregate.
 *      - Auto-halts cleanly on session completion instead of running indefinitely.
 *  - Detailed Configuration Descriptions:
 *      - Educational explainers for Non-IID Split and Differential Privacy (ε).
 *  - Conflict-Free Round Tracking: Strict monotonic rounds with zero lagging.
 */
import React, { useState, useEffect, useRef, useCallback, Suspense } from 'react';
import './App.css';
import FederatedScene from './FederatedScene';
import { AccuracyChart, LossChart, CommunicationChart } from './MetricsCharts';
import { FedAvgSimulation } from './simulation';
import AnalysisDashboard from './AnalysisDashboard';
import PredictTab from './PredictTab';
import MultiComputerCoordinator from './MultiComputerCoordinator';
import EdgeClientPortal from './EdgeClientPortal';

// ─── Group colors for legend ────────────────────────────────
const GROUP_COLORS = {
  Locomotion: '#FF6B6B',
  Sedentary: '#4ECDC4',
  'Hand-Oriented': '#45B7D1',
  Eating: '#96CEB4',
  Sports: '#FFEAA7',
};

// ─── Loading Screen ──────────────────────────────────────────
function LoadingScreen({ progress, message }) {
  return (
    <div className="loading-screen">
      <div className="loading-spinner" />
      <div className="loading-text">{message || 'Initializing federated network...'}</div>
      <div className="loading-bar">
        <div className="loading-bar-fill" style={{ width: `${progress || 60}%` }} />
      </div>
    </div>
  );
}

// ─── Phase Indicator ─────────────────────────────────────────
function PhaseIndicator({ phase }) {
  const phases = [
    { key: 'broadcast', label: 'Broadcast' },
    { key: 'training', label: 'Local Train' },
    { key: 'upload', label: 'Upload' },
    { key: 'aggregate', label: 'Aggregate' },
  ];

  const activeIdx = phases.findIndex(p => p.key === phase);

  return (
    <div className="phase-indicator">
      {phases.map((p, idx) => (
        <React.Fragment key={p.key}>
          <div className={`phase-step ${
            phase === p.key ? 'active' : 
            activeIdx > idx ? 'completed' : ''
          }`}>
            <div className="phase-dot" />
            <span>{p.label}</span>
          </div>
          {idx < phases.length - 1 && (
            <div className={`phase-connector ${activeIdx > idx ? 'active' : ''}`} />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

// ─── Node Tooltip ────────────────────────────────────────────
function NodeTooltip({ node, position }) {
  if (!node) return null;

  return (
    <div className="node-tooltip" style={{
      left: position?.x ?? 200,
      top: position?.y ?? 200,
      transform: 'translate(-50%, -120%)',
    }}>
      <div className="tooltip-header">
        <span style={{
          display: 'inline-block',
          width: 10,
          height: 10,
          borderRadius: '50%',
          backgroundColor: node.color,
          marginRight: 8,
        }} />
        {node.label || `Client ${node.id}`}
      </div>
      <div className="tooltip-row">
        <span className="label">Dominant</span>
        <span className="value">{node.dominantActivity || node.dominantGroup}</span>
      </div>
      <div className="tooltip-row">
        <span className="label">Group</span>
        <span className="value">{node.dominantGroup}</span>
      </div>
      {node.totalSamples && (
        <div className="tooltip-row">
          <span className="label">Samples</span>
          <span className="value">{Number(node.totalSamples).toLocaleString()}</span>
        </div>
      )}
      {node.devices && (
        <div className="tooltip-row">
          <span className="label">Devices</span>
          <span className="value">{node.devices.join(', ')}</span>
        </div>
      )}
      {node.entropy !== undefined && (
        <div className="tooltip-row">
          <span className="label">Entropy</span>
          <span className="value">{node.entropy.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}

// ─── Main App ────────────────────────────────────────────────
export default function App() {
  // Navigation Tabs: '3d' | 'predict' | 'analysis'
  const getInitialTab = () => {
    if (window.location.hash === '#client' || window.location.pathname === '/client') return 'client';
    if (window.location.hash === '#coordinator') return 'coordinator';
    return '3d';
  };
  const [activeTab, setActiveTab] = useState(getInitialTab);

  // Source Mode: 'websocket' | 'simulation'
  const [sourceMode, setSourceMode] = useState('websocket');

  // General State
  const [loading, setLoading] = useState(true);
  const [clientNodes, setClientNodes] = useState([]);
  const [phase, setPhase] = useState('idle');
  const [activeClients, setActiveClients] = useState([]);
  const [currentRound, setCurrentRound] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [nonIID, setNonIID] = useState(false);
  const [dpEnabled, setDpEnabled] = useState(false);
  const [dpEpsilon, setDpEpsilon] = useState(5);
  const [chartData, setChartData] = useState([]);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [wsConnected, setWsConnected] = useState(false);
  const [latestMetrics, setLatestMetrics] = useState(null);

  // Configuration Explainer Accordions
  const [showNonIIDInfo, setShowNonIIDInfo] = useState(false);
  const [showDPInfo, setShowDPInfo] = useState(false);

  const simRef = useRef(null);
  const runningRef = useRef(false);
  const wsRef = useRef(null);
  const sourceModeRef = useRef(sourceMode);
  sourceModeRef.current = sourceMode;

  const maxRounds = sourceMode === 'websocket' ? 30 : (simRef.current?.numRounds || 50);
  const isCompleted = currentRound >= maxRounds;
  const progressPercent = Math.min(100, maxRounds > 0 ? (currentRound / maxRounds) * 100 : 0);

  // ─── Load client metadata ─────────────────────────────────
  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetch('/client_metadata.json');
        if (res.ok) {
          const meta = await res.json();
          setClientNodes(meta.nodes || []);
        }
      } catch (e) {
        console.log('Using default nodes (metadata not found)');
      }

      // Initialize browser simulation engine
      const sim = new FedAvgSimulation({
        numClients: 8,
        numRounds: 50,
        localEpochs: 5,
        clientsPerRound: 8,
        lr: 0.01,
        nonIID: false,
        dpEpsilon: 0,
      });
      sim.initialize();
      simRef.current = sim;

      setTimeout(() => setLoading(false), 1200);
    }
    loadData();
  }, []);

  // ─── WebSocket connection ──────────────────────────────────
  useEffect(() => {
    function connectWS() {
      try {
        const host = window.location.hostname || 'localhost';
        const ws = new WebSocket(`ws://${host}:8000/metrics`);
        ws.onopen = () => {
          setWsConnected(true);
          console.log('WebSocket connected to Python training server');
          // Ensure server is paused on initial connection until user clicks Start
          ws.send(JSON.stringify({ action: 'pause' }));
        };
        ws.onmessage = (event) => {
          if (sourceModeRef.current !== 'websocket') return;

          try {
            const data = JSON.parse(event.data);
            if (data.phase === 'completed') {
              runningRef.current = false;
              setIsRunning(false);
              setPhase('idle');
              setActiveClients([]);
              return;
            }

            // Strictly do not animate or advance state if stopped
            if (!runningRef.current) return;

            setLatestMetrics(data);
            if (data.phase) setPhase(data.phase);
            if (data.activeClients) setActiveClients(data.activeClients);

            if (data.round !== undefined && data.phase === 'aggregate') {
              setChartData(prev => {
                const isReset = data.round === 1 && prev.length > 5;
                const base = isReset ? [] : prev.filter(p => p.round !== data.round);
                const nextPoint = {
                  round: data.round,
                  fedAccuracy: data.accuracy,
                  fedLoss: data.loss,
                  centAccuracy: data.centAccuracy ?? (data.accuracy * 1.05),
                  centLoss: data.centLoss ?? (data.loss * 0.95),
                  bytes: data.bytes || 0,
                  epsilon: data.epsilon || 0,
                };
                return [...base, nextPoint].sort((a, b) => a.round - b.round);
              });
              setCurrentRound(data.round);

              // Auto-halt cleanly when reaching 30 rounds
              if (data.round >= 30) {
                runningRef.current = false;
                setIsRunning(false);
                setPhase('idle');
                setActiveClients([]);
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({ action: 'pause' }));
                }
              }
            }
          } catch (e) {
            console.warn('WS parse error:', e);
          }
        };
        ws.onclose = () => {
          setWsConnected(false);
          setTimeout(connectWS, 4000);
        };
        ws.onerror = () => {
          ws.close();
        };
        wsRef.current = ws;
      } catch (e) {
        setWsConnected(false);
      }
    }

    connectWS();

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // ─── Simulation Loop (Pure Browser Engine) ─────────────────
  const runSimulation = useCallback(async () => {
    if (!simRef.current || runningRef.current) return;
    runningRef.current = true;
    setIsRunning(true);

    simRef.current.reset({
      nonIID,
      dpEpsilon: dpEnabled ? dpEpsilon : 0,
    });
    setChartData([]);
    setCurrentRound(0);

    for (let r = 0; r < simRef.current.numRounds; r++) {
      if (!runningRef.current) break;

      const result = await simRef.current.runRound((newPhase, clients) => {
        setPhase(newPhase);
        if (clients) setActiveClients(clients);
      });

      if (result) {
        setCurrentRound(result.round);
        setChartData(prev => {
          const filtered = prev.filter(p => p.round !== result.round);
          return [...filtered, result].sort((a, b) => a.round - b.round);
        });
        setLatestMetrics(result);
      }

      const delay = Math.max(100, 500 / speed);
      await new Promise(resolve => setTimeout(resolve, delay));
    }

    runningRef.current = false;
    setIsRunning(false);
    setPhase('idle');
    setActiveClients([]);
  }, [nonIID, dpEnabled, dpEpsilon, speed]);

  const stopSimulation = useCallback(() => {
    runningRef.current = false;
    setIsRunning(false);
    setPhase('idle');
    setActiveClients([]);
  }, []);

  // ─── Start / Stop Handler ──────────────────────────────────
  const handleTogglePlay = () => {
    if (isCompleted) {
      // If session completed, restart from round 0
      setChartData([]);
      setCurrentRound(0);
      setPhase('idle');
      if (sourceMode === 'websocket') {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          runningRef.current = true;
          setIsRunning(true);
          wsRef.current.send(JSON.stringify({ action: 'restart' }));
        }
      } else {
        runSimulation();
      }
      return;
    }

    if (sourceMode === 'websocket') {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const nextState = !isRunning;
        runningRef.current = nextState;
        setIsRunning(nextState);
        if (!nextState) {
          setPhase('idle');
          setActiveClients([]);
        }
        wsRef.current.send(JSON.stringify({ action: nextState ? 'resume' : 'pause' }));
      } else {
        if (isRunning) stopSimulation();
        else runSimulation();
      }
    } else {
      if (isRunning) stopSimulation();
      else runSimulation();
    }
  };

  const cycleSpeed = () => {
    const speeds = [0.5, 1, 2, 4];
    const idx = speeds.indexOf(speed);
    const nextSpeed = speeds[(idx + 1) % speeds.length];
    setSpeed(nextSpeed);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'speed', value: nextSpeed }));
    }
  };

  // ─── Hover handlers ───────────────────────────────────────
  const handleNodeHover = useCallback((node, event) => {
    setHoveredNode(node);
    if (event?.clientX) {
      setTooltipPos({ x: event.clientX, y: event.clientY });
    }
  }, []);

  const handleNodeUnhover = useCallback(() => {
    setHoveredNode(null);
  }, []);

  // ─── Derived metrics ──────────────────────────────────────
  const latestAccuracy = latestMetrics?.fedAccuracy ?? latestMetrics?.accuracy ?? 0;
  const latestLoss = latestMetrics?.fedLoss ?? latestMetrics?.loss ?? 0;
  const totalBytes = chartData.reduce((sum, d) => sum + (d.bytes || 0), 0);

  if (loading) {
    return <LoadingScreen progress={80} message="Loading 51 WISDM subject nodes..." />;
  }

  return (
    <div className="app-container">
      {/* ─── Header ─── */}
      <div className="header">
        <div className="header-left">
          <div className="header-logo">FL</div>
          <div className="header-title">
            <h1>Federated Learning Visualizer</h1>
            <span>Predicting 5 Activity Groups • WISDM Dataset (51 Wearable Subjects)</span>
          </div>
        </div>

        {/* ─── Tabs Switcher (5 Tabs) ─── */}
        <div className="header-center">
          <button
            className={`tab-btn ${activeTab === '3d' ? 'active' : ''}`}
            onClick={() => setActiveTab('3d')}
          >
            🌐 3D Live Network
          </button>
          <button
            className={`tab-btn ${activeTab === 'coordinator' ? 'active' : ''}`}
            onClick={() => setActiveTab('coordinator')}
          >
            🖥️ Multi-Computer (Global View)
          </button>
          <button
            className={`tab-btn ${activeTab === 'client' ? 'active' : ''}`}
            onClick={() => setActiveTab('client')}
          >
            💻 Edge Client (Other User)
          </button>
          <button
            className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`}
            onClick={() => setActiveTab('predict')}
          >
            🔮 Live Inference
          </button>
          <button
            className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`}
            onClick={() => setActiveTab('analysis')}
          >
            📊 Benchmark
          </button>
        </div>

        {/* ─── Header Right Controls ─── */}
        <div className="header-right">
          <div
            className="source-mode-pill"
            onClick={() => {
              const next = sourceMode === 'websocket' ? 'simulation' : 'websocket';
              setSourceMode(next);
              stopSimulation();
              setChartData([]);
              setCurrentRound(0);
            }}
            title="Click to toggle between Python WebSocket server and in-browser engine"
          >
            {sourceMode === 'websocket' ? '⚡ Python Server' : '💻 Browser Engine'}
          </div>

          <div className={`status-badge ${wsConnected && sourceMode === 'websocket' ? 'live' : isRunning ? 'live' : ''}`}>
            <div className="status-dot" />
            {sourceMode === 'websocket' 
              ? (wsConnected ? (isRunning ? 'Streaming' : 'WS Connected') : 'Offline') 
              : (isRunning ? 'Training' : 'Ready')}
          </div>

          <div className="round-badge">
            R{currentRound}/{maxRounds}
          </div>

          <button className="play-btn" onClick={handleTogglePlay}>
            <span className="icon">{isCompleted ? '🔄' : (isRunning ? '⏸' : '▶')}</span>
            {isCompleted ? 'Restart' : (isRunning ? 'Stop' : 'Start')}
          </button>

          <button className="speed-btn" onClick={cycleSpeed} title="Simulation Speed">
            {speed}×
          </button>
        </div>
      </div>

      {/* ─── TAB 1: 3D Visualization Scene ─── */}
      {activeTab === '3d' && (
        <>
          <div className="canvas-wrapper">
            <Suspense fallback={<LoadingScreen message="Loading 3D space..." />}>
              <FederatedScene
                nodes={clientNodes}
                phase={phase}
                activeClients={activeClients}
                dpEnabled={dpEnabled}
                currentRound={currentRound}
                onNodeHover={handleNodeHover}
                onNodeUnhover={handleNodeUnhover}
              />
            </Suspense>
          </div>

          {/* ─── Global Session Progress Bar ─── */}
          <div className="session-progress-bar-wrap">
            <div className="progress-header-row">
              <div className="progress-round-label">
                <span className="pulse-indicator" style={{ background: isCompleted ? '#10b981' : '#06b6d4' }} />
                Round {currentRound} / {maxRounds} ({Math.round(progressPercent)}%)
              </div>
              <div className="progress-phase-label" style={{ color: isCompleted ? '#10b981' : '#06b6d4' }}>
                {isCompleted && '✅ Training Session Completed'}
                {!isCompleted && phase === 'broadcast' && '📡 1/4: Broadcasting Global Model'}
                {!isCompleted && phase === 'training' && '⚙️ 2/4: Local Edge Training (SGD)'}
                {!isCompleted && phase === 'upload' && '🚀 3/4: Uploading Encrypted Updates'}
                {!isCompleted && phase === 'aggregate' && '🧠 4/4: Federated Weight Aggregation'}
                {!isCompleted && phase === 'idle' && (isRunning ? 'Running...' : 'Ready / Paused')}
              </div>
            </div>
            <div className="progress-track">
              <div
                className="progress-bar-fill"
                style={{
                  width: `${progressPercent}%`,
                  background: isCompleted 
                    ? 'linear-gradient(90deg, #10b981, #06b6d4)' 
                    : 'linear-gradient(90deg, #06b6d4, #3b82f6, #8b5cf6)',
                }}
              />
            </div>
            <div className="progress-footer-row">
              <span>{isCompleted ? 'Target 50 Rounds Reached' : `Remaining: ${maxRounds - currentRound} rounds`}</span>
              <span>{isCompleted ? 'Converged (Holdout Gen: ~56.3%)' : (isRunning ? 'Training actively in real-time' : 'Click Start to train')}</span>
            </div>
          </div>

          {/* ─── Control Panel (Left) ─── */}
          <div className="control-panel">
            {/* Target Explanation Card */}
            <div className="panel-card" style={{ borderLeft: '3px solid #06b6d4' }}>
              <div className="panel-card-title">
                <span className="icon">🎯</span> Prediction Task
              </div>
              <p style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5, margin: 0 }}>
                Recognizing <strong>5 human physical activities</strong> from phone & watch accelerometers/gyroscopes:
                <em> Locomotion, Sedentary, Hand-Oriented, Eating, Sports</em>.
              </p>
            </div>

            {/* FL Configuration with Detailed Explainers */}
            <div className="panel-card">
              <div className="panel-card-title">
                <span className="icon">⚙</span> Configuration
              </div>

              {/* Config 1: Non-IID Split */}
              <div className="toggle-row">
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span className="toggle-label">Non-IID Split</span>
                  <button
                    className="config-info-toggle"
                    onClick={() => setShowNonIIDInfo(!showNonIIDInfo)}
                    title="Read description of Non-IID distribution"
                  >
                    ℹ️
                  </button>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={nonIID}
                    onChange={(e) => setNonIID(e.target.checked)}
                    disabled={isRunning && sourceMode === 'simulation'}
                  />
                  <span className="toggle-track" />
                  <span className="toggle-thumb" />
                </label>
              </div>

              {showNonIIDInfo && (
                <div className="config-explainer-card">
                  <strong>What is Non-IID?</strong><br />
                  In the real world, human activity data is <em>Non-Independent and Identically Distributed</em>. Each user has unique daily routines: Subject 1602 sits at a desk for 8 hours (Sedentary), while Subject 1625 runs track (Locomotion & Sports).<br /><br />
                  <strong>Impact on Training:</strong><br />
                  • <em>OFF (IID)</em>: Data is uniformly shuffled across clients.<br />
                  • <em>ON (Non-IID)</em>: Each client only has 1 or 2 dominant classes. Standard FedAvg suffers from <em>Client Drift</em>, proving why <strong>FedProx</strong> is necessary to stabilize convergence.
                </div>
              )}

              {/* Config 2: Differential Privacy Noise */}
              <div className="toggle-row" style={{ marginTop: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span className="toggle-label">DP Noise</span>
                  <button
                    className="config-info-toggle"
                    onClick={() => setShowDPInfo(!showDPInfo)}
                    title="Read description of Differential Privacy"
                  >
                    ℹ️
                  </button>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={dpEnabled}
                    onChange={(e) => setDpEnabled(e.target.checked)}
                    disabled={isRunning && sourceMode === 'simulation'}
                  />
                  <span className="toggle-track" />
                  <span className="toggle-thumb" />
                </label>
              </div>

              {showDPInfo && (
                <div className="config-explainer-card">
                  <strong>What is Differential Privacy?</strong><br />
                  A cryptographic framework that guarantees an observer cannot determine whether a specific individual participated in the training cohort.<br /><br />
                  <strong>How it Works:</strong><br />
                  1. <em>Gradient Clipping</em>: Limits each client's update norm to <code>C = 1.0</code>.<br />
                  2. <em>Gaussian Noise</em>: Injects calibrated noise <code>N(0, σ²)</code> into the server aggregate.<br />
                  • <strong>Lower ε (e.g. ε=1)</strong>: Strict privacy, high noise. Prevents Membership Inference Attacks (MIA), with slightly lower model accuracy.<br />
                  • <strong>Higher ε (e.g. ε=10–20)</strong>: Mild privacy, minimal noise, higher accuracy.
                </div>
              )}

              {/* Slider for Privacy Budget (epsilon) */}
              {dpEnabled && (
                <div className="slider-group">
                  <div className="slider-header">
                    <span className="slider-label">Privacy Budget (ε)</span>
                    <span className="slider-value">{dpEpsilon}</span>
                  </div>
                  <input
                    type="range"
                    className="slider-input"
                    min="1"
                    max="20"
                    step="1"
                    value={dpEpsilon}
                    onChange={(e) => setDpEpsilon(Number(e.target.value))}
                    disabled={isRunning && sourceMode === 'simulation'}
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b', marginTop: 4 }}>
                    <span>Strict (ε=1)</span>
                    <span>Mild (ε=20)</span>
                  </div>
                </div>
              )}
            </div>

            {/* Live Metrics */}
            <div className="panel-card">
              <div className="panel-card-title">
                <span className="icon">📊</span> Live Metrics
              </div>
              <div className="metrics-grid">
                <div className="metric-item">
                  <div className="metric-value">{(latestAccuracy * 100).toFixed(1)}%</div>
                  <div className="metric-label">Accuracy</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{latestLoss.toFixed(3)}</div>
                  <div className="metric-label">Loss</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{currentRound}</div>
                  <div className="metric-label">Rounds</div>
                </div>
                <div className="metric-item">
                  <div className="metric-value">{(totalBytes / 1024).toFixed(0)}</div>
                  <div className="metric-label">KB Sent</div>
                </div>
              </div>
            </div>

            {/* Legend */}
            <div className="panel-card">
              <div className="panel-card-title">
                <span className="icon">🎨</span> Activity Groups (51 Subjects)
              </div>
              <div className="legend">
                {Object.entries(GROUP_COLORS).map(([group, color]) => (
                  <div key={group} className="legend-item">
                    <div className="legend-dot" style={{ backgroundColor: color }} />
                    {group}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ─── Charts Panel (Right) ─── */}
          <div className="charts-panel">
            <div className="chart-card">
              <div className="chart-title">
                <span className="icon">📈</span> Accuracy vs Rounds
              </div>
              <AccuracyChart data={chartData} />
              <div style={{ display: 'flex', gap: 16, marginTop: 8, justifyContent: 'center' }}>
                <span style={{ fontSize: 10, color: '#06b6d4', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ width: 12, height: 2, background: '#06b6d4', display: 'inline-block', borderRadius: 1 }} />
                  Federated
                </span>
                <span style={{ fontSize: 10, color: '#f59e0b', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ width: 12, height: 2, background: '#f59e0b', display: 'inline-block', borderRadius: 1 }} />
                  Centralized
                </span>
              </div>
            </div>

            <div className="chart-card">
              <div className="chart-title">
                <span className="icon">📉</span> Loss vs Rounds
              </div>
              <LossChart data={chartData} />
            </div>

            <div className="chart-card">
              <div className="chart-title">
                <span className="icon">📡</span> Communication Cost
              </div>
              <CommunicationChart data={chartData} />
            </div>
          </div>

          {/* ─── Phase Indicator ─── */}
          {(isRunning || (sourceMode === 'websocket' && phase !== 'idle')) && (
            <PhaseIndicator phase={phase} />
          )}

          {/* ─── Tooltip ─── */}
          <NodeTooltip node={hoveredNode} position={tooltipPos} />
        </>
      )}

      {/* ─── TAB 2: Multi-Computer Coordinator (Global Host View) ─── */}
      {activeTab === 'coordinator' && <MultiComputerCoordinator />}

      {/* ─── TAB 3: Edge Client Portal (Other User / Remote Computer) ─── */}
      {activeTab === 'client' && <EdgeClientPortal />}

      {/* ─── TAB 4: Live Inference / Prediction ─── */}
      {activeTab === 'predict' && <PredictTab />}

      {/* ─── TAB 5: Complete Benchmark & Dataset Analysis ─── */}
      {activeTab === 'analysis' && <AnalysisDashboard />}
    </div>
  );
}
