"""
WISDM Dataset - Comprehensive Data Analysis & Feature Engineering
=================================================================
Loads raw phone/watch accelerometer+gyroscope data for all 51 subjects,
windows the signals, extracts features, and produces:
  - Per-client sample counts
  - Per-client class distributions
  - Feature-engineered dataset for FL training
  - Visualization-ready JSON metadata
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURATION ───────────────────────────────────────────────
RAW_DIR = r"d:\Education\DPSA_proj\wisdm-dataset\wisdm-dataset\raw"
OUTPUT_DIR = r"d:\Education\DPSA_proj\analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTIVITY_MAP = {
    'A': 'Walking', 'B': 'Jogging', 'C': 'Stairs',
    'D': 'Sitting', 'E': 'Standing', 'F': 'Typing',
    'G': 'Brushing Teeth', 'H': 'Eating Soup', 'I': 'Eating Chips',
    'J': 'Eating Pasta', 'K': 'Drinking', 'L': 'Eating Sandwich',
    'M': 'Kicking', 'O': 'Catching', 'P': 'Dribbling',
    'Q': 'Writing', 'R': 'Clapping', 'S': 'Folding'
}

# 6-group merge for simpler classification
ACTIVITY_GROUPS = {
    'A': 'Locomotion', 'B': 'Locomotion', 'C': 'Locomotion',
    'D': 'Sedentary', 'E': 'Sedentary',
    'F': 'Hand-Oriented', 'G': 'Hand-Oriented', 'Q': 'Hand-Oriented', 'R': 'Hand-Oriented', 'S': 'Hand-Oriented',
    'H': 'Eating', 'I': 'Eating', 'J': 'Eating', 'K': 'Eating', 'L': 'Eating',
    'M': 'Sports', 'O': 'Sports', 'P': 'Sports',
}

GROUP_COLORS = {
    'Locomotion': '#FF6B6B', 'Sedentary': '#4ECDC4',
    'Hand-Oriented': '#45B7D1', 'Eating': '#96CEB4',
    'Sports': '#FFEAA7',
}

SAMPLE_RATE = 20  # Hz
WINDOW_SEC = 10
WINDOW_SIZE = SAMPLE_RATE * WINDOW_SEC  # 200 samples
OVERLAP = 0.5
STEP = int(WINDOW_SIZE * (1 - OVERLAP))

# ─── DATA LOADING ────────────────────────────────────────────────
def parse_raw_file(filepath):
    """Parse a WISDM raw sensor file."""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';').strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                continue
            try:
                sid = int(parts[0].strip())
                act = parts[1].strip()
                ts = int(parts[2].strip())
                x = float(parts[3].strip())
                y = float(parts[4].strip())
                z = float(parts[5].strip().rstrip(';'))
                rows.append((sid, act, ts, x, y, z))
            except (ValueError, IndexError):
                continue
    if not rows:
        return pd.DataFrame(columns=['subject', 'activity', 'timestamp', 'x', 'y', 'z'])
    df = pd.DataFrame(rows, columns=['subject', 'activity', 'timestamp', 'x', 'y', 'z'])
    return df

def load_all_raw_data():
    """Load all raw sensor data from phone and watch, accel and gyro."""
    all_data = {}
    for device in ['phone', 'watch']:
        for sensor in ['accel', 'gyro']:
            folder = os.path.join(RAW_DIR, device, sensor)
            if not os.path.exists(folder):
                continue
            pattern = os.path.join(folder, f"data_*_{sensor}_{device}.txt")
            files = sorted(glob.glob(pattern))
            print(f"  Loading {device}/{sensor}: {len(files)} files")
            for fpath in files:
                sid = int(os.path.basename(fpath).split('_')[1])
                df = parse_raw_file(fpath)
                if df.empty:
                    continue
                key = (sid, device, sensor)
                all_data[key] = df
    return all_data

# ─── FEATURE EXTRACTION ─────────────────────────────────────────
def extract_window_features(window_xyz):
    """Extract statistical and FFT features from a (WINDOW_SIZE, 3) array."""
    features = {}
    axes = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        vals = window_xyz[:, i]
        features[f'{ax}_mean'] = np.mean(vals)
        features[f'{ax}_std'] = np.std(vals)
        features[f'{ax}_min'] = np.min(vals)
        features[f'{ax}_max'] = np.max(vals)
        features[f'{ax}_median'] = np.median(vals)
        features[f'{ax}_iqr'] = np.percentile(vals, 75) - np.percentile(vals, 25)
        features[f'{ax}_mad'] = np.mean(np.abs(vals - np.mean(vals)))
        features[f'{ax}_rms'] = np.sqrt(np.mean(vals**2))
        features[f'{ax}_skew'] = float(pd.Series(vals).skew())
        features[f'{ax}_kurt'] = float(pd.Series(vals).kurtosis())
        # Zero crossing rate
        features[f'{ax}_zcr'] = np.sum(np.diff(np.sign(vals - np.mean(vals))) != 0) / len(vals)
        # FFT features (top 5 magnitudes and frequencies)
        fft_vals = np.abs(rfft(vals))
        fft_freqs = rfftfreq(len(vals), d=1.0/SAMPLE_RATE)
        top_idx = np.argsort(fft_vals)[-5:][::-1]
        for j in range(5):
            features[f'{ax}_fft_mag_{j}'] = fft_vals[top_idx[j]]
            features[f'{ax}_fft_freq_{j}'] = fft_freqs[top_idx[j]]
        # Spectral energy
        features[f'{ax}_spectral_energy'] = np.sum(fft_vals**2)

    # Cross-axis features
    mag = np.sqrt(np.sum(window_xyz**2, axis=1))
    features['magnitude_mean'] = np.mean(mag)
    features['magnitude_std'] = np.std(mag)
    # Correlation between axes
    if np.std(window_xyz[:, 0]) > 0 and np.std(window_xyz[:, 1]) > 0:
        features['xy_corr'] = np.corrcoef(window_xyz[:, 0], window_xyz[:, 1])[0, 1]
    else:
        features['xy_corr'] = 0
    if np.std(window_xyz[:, 0]) > 0 and np.std(window_xyz[:, 2]) > 0:
        features['xz_corr'] = np.corrcoef(window_xyz[:, 0], window_xyz[:, 2])[0, 1]
    else:
        features['xz_corr'] = 0
    if np.std(window_xyz[:, 1]) > 0 and np.std(window_xyz[:, 2]) > 0:
        features['yz_corr'] = np.corrcoef(window_xyz[:, 1], window_xyz[:, 2])[0, 1]
    else:
        features['yz_corr'] = 0

    return features

def window_and_extract(df, device, sensor):
    """Window a DataFrame and extract features per window."""
    records = []
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    for act_label in df_sorted['activity'].unique():
        act_df = df_sorted[df_sorted['activity'] == act_label]
        xyz = act_df[['x', 'y', 'z']].values

        for start in range(0, len(xyz) - WINDOW_SIZE + 1, STEP):
            window = xyz[start:start + WINDOW_SIZE]
            feats = extract_window_features(window)
            feats['activity'] = act_label
            feats['activity_name'] = ACTIVITY_MAP.get(act_label, act_label)
            feats['activity_group'] = ACTIVITY_GROUPS.get(act_label, 'Other')
            feats['subject'] = int(df_sorted['subject'].iloc[0])
            feats['device'] = device
            feats['sensor'] = sensor
            records.append(feats)

    return records

# ─── ANALYSIS FUNCTIONS ─────────────────────────────────────────
def analyze_client_distributions(all_data):
    """Analyze per-client sample and class distributions."""
    client_stats = defaultdict(lambda: {
        'total_samples': 0,
        'activities': Counter(),
        'activity_groups': Counter(),
        'devices': set(),
        'sensors': set()
    })

    for (sid, device, sensor), df in all_data.items():
        stats = client_stats[sid]
        stats['total_samples'] += len(df)
        stats['devices'].add(device)
        stats['sensors'].add(sensor)
        for act, count in df['activity'].value_counts().items():
            stats['activities'][act] += count
            stats['activity_groups'][ACTIVITY_GROUPS.get(act, 'Other')] += count

    return dict(client_stats)

def compute_non_iid_metrics(client_stats):
    """Compute non-IID metrics across clients."""
    all_activities = set()
    for sid, stats in client_stats.items():
        all_activities.update(stats['activities'].keys())

    all_activities = sorted(all_activities)
    n_clients = len(client_stats)

    # Build distribution matrix
    dist_matrix = np.zeros((n_clients, len(all_activities)))
    sids = sorted(client_stats.keys())
    for i, sid in enumerate(sids):
        total = sum(client_stats[sid]['activities'].values())
        for j, act in enumerate(all_activities):
            dist_matrix[i, j] = client_stats[sid]['activities'].get(act, 0) / max(total, 1)

    # Variance of class proportions
    class_variance = np.var(dist_matrix, axis=0)

    # Entropy per client
    entropies = []
    for i in range(n_clients):
        p = dist_matrix[i]
        p = p[p > 0]
        entropies.append(-np.sum(p * np.log2(p)))

    return {
        'class_variance': class_variance.tolist(),
        'mean_entropy': float(np.mean(entropies)),
        'std_entropy': float(np.std(entropies)),
        'min_entropy': float(np.min(entropies)),
        'max_entropy': float(np.max(entropies)),
        'activities': all_activities,
        'dist_matrix': dist_matrix,
        'sids': sids
    }

# ─── VISUALIZATION ───────────────────────────────────────────────
def plot_samples_per_client(client_stats):
    """Plot samples per client with color-coded dominant activity."""
    fig, ax = plt.subplots(figsize=(18, 7))

    sids = sorted(client_stats.keys())
    sample_counts = []
    dominant_colors = []

    for sid in sids:
        total = sum(client_stats[sid]['activities'].values())
        sample_counts.append(total)
        group_counts = client_stats[sid]['activity_groups']
        dom_group = max(group_counts, key=group_counts.get)
        dominant_colors.append(GROUP_COLORS.get(dom_group, '#999'))

    x = np.arange(len(sids))
    ax.bar(x, sample_counts, color=dominant_colors, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Client (Subject ID)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Sensor Readings', fontsize=12, fontweight='bold')
    ax.set_title('Samples Per Client - Colored by Dominant Activity Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(s) for s in sids[::2]], rotation=45, fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'samples_per_client.png'), dpi=150)
    plt.close()
    print("  Saved samples_per_client.png")

def plot_class_distribution_heatmap(noniid_metrics):
    """Plot class distribution heatmap across clients."""
    fig, ax = plt.subplots(figsize=(16, 12))

    im = ax.imshow(noniid_metrics['dist_matrix'], aspect='auto', cmap='YlOrRd')
    ax.set_xlabel('Activity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Client (Subject ID)', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution Heatmap Across Clients (Non-IID Analysis)', fontsize=14, fontweight='bold')

    act_labels = [ACTIVITY_MAP.get(a, a) for a in noniid_metrics['activities']]
    ax.set_xticks(np.arange(len(act_labels)))
    ax.set_xticklabels(act_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(noniid_metrics['sids']))[::2])
    ax.set_yticklabels([str(s) for s in noniid_metrics['sids'][::2]], fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Samples', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution_heatmap.png'), dpi=150)
    plt.close()
    print("  Saved class_distribution_heatmap.png")

def plot_activity_group_pie(client_stats):
    """Plot overall activity group distribution."""
    group_totals = Counter()
    for sid, stats in client_stats.items():
        for g, c in stats['activity_groups'].items():
            group_totals[g] += c

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    labels = list(group_totals.keys())
    sizes = list(group_totals.values())
    colors = [GROUP_COLORS.get(l, '#999') for l in labels]
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, pctdistance=0.85,
                wedgeprops=dict(width=0.5, edgecolor='white'))
    axes[0].set_title('Overall Activity Group Distribution', fontsize=13, fontweight='bold')

    sids = sorted(client_stats.keys())
    groups = sorted(GROUP_COLORS.keys())
    bottom = np.zeros(len(sids))
    for group in groups:
        vals = []
        for sid in sids:
            total = sum(client_stats[sid]['activities'].values())
            gcount = client_stats[sid]['activity_groups'].get(group, 0)
            vals.append(gcount / max(total, 1))
        vals = np.array(vals)
        axes[1].bar(range(len(sids)), vals, bottom=bottom,
                    color=GROUP_COLORS[group], label=group, edgecolor='white', linewidth=0.3)
        bottom += vals

    axes[1].set_xlabel('Client Index', fontsize=11)
    axes[1].set_ylabel('Proportion', fontsize=11)
    axes[1].set_title('Per-Client Activity Group Proportions', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].set_xlim(-0.5, len(sids) - 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'activity_group_distribution.png'), dpi=150)
    plt.close()
    print("  Saved activity_group_distribution.png")

def plot_sensor_signal_samples(all_data):
    """Plot raw sensor signals for a few activities from one subject."""
    sid = 1600
    key = (sid, 'phone', 'accel')
    if key not in all_data:
        print("  Subject 1600 phone accel not found for signal plot")
        return

    df = all_data[key]
    activities_to_show = ['A', 'B', 'D', 'M']
    activities_to_show = [a for a in activities_to_show if a in df['activity'].unique()]

    fig, axes = plt.subplots(len(activities_to_show), 1, figsize=(16, 3 * len(activities_to_show)))
    if len(activities_to_show) == 1:
        axes = [axes]

    for idx, act in enumerate(activities_to_show):
        act_df = df[df['activity'] == act].sort_values('timestamp').head(400)
        t = np.arange(len(act_df)) / SAMPLE_RATE

        axes[idx].plot(t, act_df['x'].values, label='X', alpha=0.8, linewidth=0.8)
        axes[idx].plot(t, act_df['y'].values, label='Y', alpha=0.8, linewidth=0.8)
        axes[idx].plot(t, act_df['z'].values, label='Z', alpha=0.8, linewidth=0.8)
        axes[idx].set_title(f'{ACTIVITY_MAP[act]} (Subject {sid}, Phone Accel)', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Acceleration', fontsize=9)
        axes[idx].legend(fontsize=8, loc='upper right')
        axes[idx].grid(alpha=0.3)

    axes[-1].set_xlabel('Time (seconds)', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sensor_signal_samples.png'), dpi=150)
    plt.close()
    print("  Saved sensor_signal_samples.png")

def plot_device_heterogeneity(all_data):
    """Compare phone vs watch sensor distributions."""
    phone_mags = []
    watch_mags = []

    for (sid, device, sensor), df in all_data.items():
        if sensor != 'accel':
            continue
        mag = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
        sample = mag.sample(min(500, len(mag)), random_state=42).values
        if device == 'phone':
            phone_mags.extend(sample)
        else:
            watch_mags.extend(sample)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(phone_mags, bins=80, alpha=0.7, color='#FF6B6B', label='Phone', density=True)
    axes[0].hist(watch_mags, bins=80, alpha=0.7, color='#4ECDC4', label='Watch', density=True)
    axes[0].set_title('Accel Magnitude Distribution: Phone vs Watch', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Magnitude (m/s²)')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    phone_stds = []
    watch_stds = []
    for (sid, device, sensor), df in all_data.items():
        if sensor != 'accel':
            continue
        std_val = df[['x', 'y', 'z']].std().mean()
        if device == 'phone':
            phone_stds.append(std_val)
        else:
            watch_stds.append(std_val)

    bp = axes[1].boxplot([phone_stds, watch_stds], labels=['Phone', 'Watch'],
                          patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#FF6B6B')
    bp['boxes'][1].set_facecolor('#4ECDC4')
    axes[1].set_title('Sensor Variance: Phone vs Watch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Std Dev Across Axes')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'device_heterogeneity.png'), dpi=150)
    plt.close()
    print("  Saved device_heterogeneity.png")

def plot_feature_importance_preview(feature_df):
    """Quick feature correlation preview."""
    from sklearn.ensemble import RandomForestClassifier

    feat_cols = [c for c in feature_df.columns if c not in
                 ['activity', 'activity_name', 'activity_group', 'subject', 'device', 'sensor']]

    X = feature_df[feat_cols].values
    y = LabelEncoder().fit_transform(feature_df['activity_group'].values)
    X = np.nan_to_num(X, nan=0.0)

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = [feat_cols[i] for i in top_idx]
    top_importances = importances[top_idx]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_importances[::-1], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (Random Forest)', fontsize=11)
    ax.set_title('Top 20 Features for Activity Group Classification', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    plt.close()
    print("  Saved feature_importance.png")

# ─── GENERATE VISUALIZATION METADATA ─────────────────────────────
def generate_vis_metadata(client_stats, noniid_metrics):
    """Generate JSON metadata for the React Three Fiber visualization."""
    nodes = []
    sids = sorted(client_stats.keys())

    for sid in sids:
        stats = client_stats[sid]
        total = sum(stats['activities'].values())
        dom_group = max(stats['activity_groups'], key=stats['activity_groups'].get)
        dom_act = max(stats['activities'], key=stats['activities'].get)

        nodes.append({
            'id': sid,
            'label': f'Subject {sid}',
            'totalSamples': total,
            'dominantGroup': dom_group,
            'dominantActivity': ACTIVITY_MAP.get(dom_act, dom_act),
            'dominantActivityCode': dom_act,
            'color': GROUP_COLORS.get(dom_group, '#999'),
            'devices': list(stats['devices']),
            'numActivities': len(stats['activities']),
            'activityDistribution': {
                ACTIVITY_MAP.get(k, k): v for k, v in stats['activities'].items()
            },
            'groupDistribution': dict(stats['activity_groups']),
            'entropy': float(-sum(
                (v/total) * np.log2(v/total) for v in stats['activities'].values() if v > 0
            )) if total > 0 else 0
        })

    metadata = {
        'totalClients': len(sids),
        'totalActivities': 18,
        'activityGroups': list(GROUP_COLORS.keys()),
        'groupColors': GROUP_COLORS,
        'activityMap': ACTIVITY_MAP,
        'nodes': nodes,
        'nonIidMetrics': {
            'meanEntropy': noniid_metrics['mean_entropy'],
            'stdEntropy': noniid_metrics['std_entropy'],
            'minEntropy': noniid_metrics['min_entropy'],
            'maxEntropy': noniid_metrics['max_entropy'],
        },
        'sampleRate': SAMPLE_RATE,
        'windowSize': WINDOW_SIZE,
        'windowSec': WINDOW_SEC,
        'overlap': OVERLAP,
    }

    with open(os.path.join(OUTPUT_DIR, 'client_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  Saved client_metadata.json")
    return metadata

# ─── MAIN ────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  WISDM Dataset - Comprehensive Analysis")
    print("=" * 70)

    # 1. Load all raw data
    print("\n[1/7] Loading raw sensor data...")
    all_data = load_all_raw_data()
    print(f"  Total data streams loaded: {len(all_data)}")

    total_readings = sum(len(df) for df in all_data.values())
    subjects = sorted(set(k[0] for k in all_data.keys()))
    print(f"  Total sensor readings: {total_readings:,}")
    print(f"  Subjects: {len(subjects)} ({min(subjects)}-{max(subjects)})")

    # 2. Client distributions
    print("\n[2/7] Analyzing client distributions...")
    client_stats = analyze_client_distributions(all_data)

    # 3. Non-IID analysis
    print("\n[3/7] Computing non-IID metrics...")
    noniid_metrics = compute_non_iid_metrics(client_stats)
    print(f"  Mean class entropy: {noniid_metrics['mean_entropy']:.3f}")
    print(f"  Entropy range: [{noniid_metrics['min_entropy']:.3f}, {noniid_metrics['max_entropy']:.3f}]")

    # 4. Plots
    print("\n[4/7] Generating analysis plots...")
    plot_samples_per_client(client_stats)
    plot_class_distribution_heatmap(noniid_metrics)
    plot_activity_group_pie(client_stats)
    plot_sensor_signal_samples(all_data)
    plot_device_heterogeneity(all_data)

    # 5. Feature extraction (phone accel only for speed)
    print("\n[5/7] Extracting windowed features (phone accel)...")
    all_features = []
    for sid in subjects:
        key = (sid, 'phone', 'accel')
        if key in all_data:
            feats = window_and_extract(all_data[key], 'phone', 'accel')
            all_features.extend(feats)
            if sid % 10 == 0:
                print(f"  Subject {sid}: {len(feats)} windows")

    feature_df = pd.DataFrame(all_features)
    print(f"  Total feature windows: {len(feature_df):,}")
    print(f"  Feature columns: {len([c for c in feature_df.columns if c not in ['activity', 'activity_name', 'activity_group', 'subject', 'device', 'sensor']])}")

    # 6. Feature importance
    print("\n[6/7] Computing feature importance...")
    plot_feature_importance_preview(feature_df)

    # 7. Save outputs
    print("\n[7/7] Saving processed data and metadata...")
    feature_df.to_csv(os.path.join(OUTPUT_DIR, 'features_phone_accel.csv'), index=False)
    print(f"  Saved features_phone_accel.csv ({len(feature_df):,} rows)")

    metadata = generate_vis_metadata(client_stats, noniid_metrics)

    # Summary statistics JSON
    summary = {
        'dataset': 'WISDM v2',
        'totalSubjects': len(subjects),
        'subjectRange': [min(subjects), max(subjects)],
        'totalReadings': total_readings,
        'sampleRateHz': SAMPLE_RATE,
        'windowSizeSec': WINDOW_SEC,
        'overlapPct': int(OVERLAP * 100),
        'totalFeatureWindows': len(feature_df),
        'numFeatures': len([c for c in feature_df.columns if c not in
                           ['activity', 'activity_name', 'activity_group', 'subject', 'device', 'sensor']]),
        'activities': ACTIVITY_MAP,
        'activityGroups': {k: v for k, v in ACTIVITY_GROUPS.items()},
        'groupColors': GROUP_COLORS,
        'perClientStats': {
            str(sid): {
                'totalSamples': sum(client_stats[sid]['activities'].values()),
                'numActivities': len(client_stats[sid]['activities']),
                'devices': list(client_stats[sid]['devices']),
            }
            for sid in sorted(client_stats.keys())
        }
    }
    with open(os.path.join(OUTPUT_DIR, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Saved analysis_summary.json")

    # Print report
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  {len(subjects)} clients (subjects)")
    print(f"  {total_readings:,} total sensor readings")
    print(f"  {len(feature_df):,} windowed feature vectors")
    print(f"  18 activities -> 5 groups")
    print(f"  Devices: phone + watch x (accel + gyro)")
    print(f"\n  Outputs saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
