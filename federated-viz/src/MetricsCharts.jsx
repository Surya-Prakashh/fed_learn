/**
 * MetricsCharts.jsx — Live Training Charts
 * ==========================================
 * Real-time accuracy and loss plots for federated vs centralized training.
 */
import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Area, AreaChart, Legend
} from 'recharts';

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div style={{
      background: 'rgba(15, 23, 42, 0.95)',
      border: '1px solid rgba(6, 182, 212, 0.3)',
      borderRadius: '8px',
      padding: '10px 14px',
      backdropFilter: 'blur(12px)',
      fontSize: '12px',
      fontFamily: 'Inter, sans-serif',
    }}>
      <p style={{ color: '#94a3b8', marginBottom: '6px', fontWeight: 600 }}>
        Round {label}
      </p>
      {payload.map((entry, idx) => (
        <p key={idx} style={{ color: entry.color, margin: '2px 0' }}>
          {entry.name}: <span style={{ fontFamily: 'JetBrains Mono', fontWeight: 600 }}>
            {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
          </span>
        </p>
      ))}
    </div>
  );
};

export function AccuracyChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div style={{ height: 160, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: '#64748b', fontSize: 12 }}>Waiting for training data...</span>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="fedGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="centGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
        <XAxis
          dataKey="round"
          stroke="#475569"
          fontSize={10}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          domain={[0, 1]}
          stroke="#475569"
          fontSize={10}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
        />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="monotone"
          dataKey="fedAccuracy"
          stroke="#06b6d4"
          strokeWidth={2}
          fill="url(#fedGrad)"
          name="Federated"
          dot={false}
          activeDot={{ r: 4, fill: '#06b6d4' }}
        />
        <Area
          type="monotone"
          dataKey="centAccuracy"
          stroke="#f59e0b"
          strokeWidth={2}
          fill="url(#centGrad)"
          name="Centralized"
          dot={false}
          activeDot={{ r: 4, fill: '#f59e0b' }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function LossChart({ data }) {
  if (!data || data.length === 0) {
    return (
      <div style={{ height: 140, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ color: '#64748b', fontSize: 12 }}>Waiting for training data...</span>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={140}>
      <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
        <XAxis
          dataKey="round"
          stroke="#475569"
          fontSize={10}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          stroke="#475569"
          fontSize={10}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => v.toFixed(1)}
        />
        <Tooltip content={<CustomTooltip />} />
        <Line
          type="monotone"
          dataKey="fedLoss"
          stroke="#ef4444"
          strokeWidth={2}
          name="Fed Loss"
          dot={false}
          activeDot={{ r: 4, fill: '#ef4444' }}
        />
        <Line
          type="monotone"
          dataKey="centLoss"
          stroke="#a855f7"
          strokeWidth={2}
          name="Cent Loss"
          dot={false}
          activeDot={{ r: 4, fill: '#a855f7' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function CommunicationChart({ data }) {
  if (!data || data.length === 0) return null;

  const commData = data.map(d => ({
    round: d.round,
    kbytes: (d.bytes / 1024).toFixed(1),
  }));

  return (
    <ResponsiveContainer width="100%" height={100}>
      <AreaChart data={commData} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="commGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
        <XAxis dataKey="round" stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
        <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="monotone"
          dataKey="kbytes"
          stroke="#10b981"
          strokeWidth={2}
          fill="url(#commGrad)"
          name="KB/Round"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
