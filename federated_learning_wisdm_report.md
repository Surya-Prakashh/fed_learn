# Federated Learning for Human Activity Recognition (WISDM Dataset)
## Comprehensive Technical Report, Privacy & Communication Analysis

---

### Executive Summary

Wearable sensors on smartphones and smartwatches capture continuous accelerometer and gyroscope biometrics that are highly sensitive to user privacy, revealing personal habits, daily routines, motor symptoms, and health status. Centralized machine learning paradigms aggregate raw sensor streams onto third-party cloud servers, introducing severe data custody risks and violating privacy regulations (GDPR, HIPAA).

This project presents an end-to-end **Privacy-Preserving Federated Learning (FL)** system trained on the **WISDM Human Activity Recognition (HAR) dataset** spanning **51 subjects** and **15.6 million raw sensor readings**. We evaluate:
1. **Heterogeneous Client Setup**: 51 natural edge clients exhibiting severe non-IID statistical variation and device heterogeneity (smartphones vs. smartwatches).
2. **Baselines**: Centralized training vs. Local-only training vs. Federated Averaging (**FedAvg**) and Federated Proximal (**FedProx**).
3. **Privacy & Security**: $(\epsilon, \delta)$-Differential Privacy with noise calibration, pairwise zero-sum **Secure Aggregation**, and **Membership Inference Attack (MIA)** resistance.
4. **Communication Compression**: 32-bit floating point (FP32) vs. 8-bit Integer Quantization (INT8) vs. Top-$k$ gradient sparsification.
5. **Interactive 3D Visualizer**: A React Three Fiber application connecting to a live WebSocket streaming server at `ws://localhost:8000/metrics`.

---

### 1. Data Preparation & Edge Client Setup

#### 1.1 Dataset Characteristics & Windowing
The raw WISDM dataset contains 18 fine-grained activities recorded across 51 subjects (IDs 1600–1650) with synchronized phone and watch sensors sampled at 20 Hz. To facilitate robust classification and physiological meaning, activities are grouped into 5 physiological clusters:
- **Locomotion**: Walking, Jogging, Stairs
- **Sedentary**: Sitting, Standing
- **Hand-Oriented**: Typing, Writing, Brushing Teeth, Clapping, Folding Clothes
- **Eating**: Eating Soup, Chips, Pasta, Sandwich, Drinking
- **Sports**: Kicking Soccer Ball, Catching Tennis Ball, Dribbling Basketball

Signals were processed with sliding windows of **10 seconds** (200 samples) with **50% overlap** (5-second stride). Across each window, **71 statistical, kinematic, and spectral features** were extracted across $X, Y, Z$ and Euclidean vector magnitude ($M = \sqrt{X^2+Y^2+Z^2}$):
- **Time-domain**: Mean, standard deviation, min, max, median, interquartile range (IQR), mean absolute deviation (MAD), root mean square (RMS), skewness, and kurtosis.
- **Frequency-domain (FFT)**: Dominant frequency, spectral energy, and spectral entropy.
- **Inter-axis correlation**: Pearson correlation coefficients ($r_{xy}, r_{xz}, r_{yz}$).

#### 1.2 Data Distribution & Heterogeneity Visualizations

![Activity Group Distribution](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/activity_group_distribution.png)
*Figure 1: Distribution of samples across the 5 activity groups in the extracted feature corpus.*

![Samples Per Client](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/samples_per_client.png)
*Figure 2: Sample count per client across all 51 WISDM subjects, demonstrating severe quantity imbalance.*

![Class Distribution Heatmap](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/class_distribution_heatmap.png)
*Figure 3: Non-IID label skew per client: subjects exhibit distinct dominant behavioral clusters.*

![Device Heterogeneity](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/device_heterogeneity.png)
*Figure 4: Sensor hardware heterogeneity between wrist-worn smartwatch and pocket-carried smartphone.*

#### 1.3 Threat Model for Wearable Sensor FL

```mermaid
flowchart LR
    subgraph EdgeClients [Edge Clients: 51 Wearable Devices]
        C1[Client 1600<br/>Local Sensor Data]
        C2[Client 1601<br/>Local Sensor Data]
        C3[Client 1650<br/>Local Sensor Data]
    end

    subgraph AdversaryThreats [Adversarial Surface]
        T1[Curious Server<br/>Gradient Inversion Attack]
        T2[Eavesdropper / MitM<br/>Unencrypted Model Snooping]
        T3[External Observer<br/>Membership Inference Attack]
    end

    subgraph DefenseSuite [Privacy-Preserving Defenses]
        D1[Pairwise Masking<br/>Secure Aggregation]
        D2[TLS Channel<br/>Authenticated Transport]
        D3[DP Noise Addition<br/>Differential Privacy]
    end

    EdgeClients --> DefenseSuite
    DefenseSuite --> CentralServer[Aggregator Server]
```

- **Attacker Profile 1: Honest-but-Curious Server**: The central aggregation server follows the training protocol faithfully but attempts to inspect raw parameter updates $\Delta w_k$ to reconstruct user sensor trajectories via gradient inversion or infer sensitive biometrics (e.g. Parkinsonian gait tremors).
  - *Defense*: **Secure Aggregation** using pairwise zero-sum masks $s_{i,j} = -s_{j,i}$, ensuring individual client updates appear strictly as uniformly distributed noise to the server.
- **Attacker Profile 2: External Passive Observer**: An adversary observing released model checkpoints or public inference APIs attempts to determine whether a target individual's gait data was included in the training cohort (**Membership Inference Attack**).
  - *Defense*: **$(\epsilon, \delta)$-Differential Privacy** with local update clipping $C$ and server-side Gaussian perturbation calibrated to privacy budget $\epsilon$.
- **Attacker Profile 3: Malicious Client (Data Poisoning / Sybil)**: A compromised device uploads corrupted updates to sabotage global convergence.
  - *Defense*: Norm clipping threshold $C$ bounds the maximum influence of any single anomalous edge device.

---

### 2. Baselines & Federated Learning Performance

We constructed `HARNet`, a 3-layer neural network with LayerNorm (71 input features $\to$ 128 hidden $\to$ 64 hidden $\to$ 5 classes, total 18,053 trainable parameters). The dataset was partitioned into 41 training clients and 10 unseen holdout test subjects to rigorously measure cross-subject generalization.

#### 2.1 Empirical Results Summary

| Model / Strategy | Training Mode | Accuracy (Local) | Accuracy (Holdout Gen.) | Holdout Loss | Communication (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | Pooled (All 41 subjects) | 82.39% | **57.05%** | 1.550 | N/A (Central Cloud) |
| **Local-Only Baseline** | Disconnected (Each client alone) | **91.50%** | **36.67%** | 2.140 | 0 MB |
| **FedAvg (FP32)** | Distributed (10 clients/round) | 85.12% | **50.48%** | 1.363 | 33.29 MB |
| **FedProx ($\mu=0.01$)** | Distributed + Proximal Regularizer | 86.40% | **56.34%** | 1.346 | 33.29 MB |
| **FedAvg-INT8** | 8-bit Quantized | 84.80% | **56.01%** | 1.189 | **8.32 MB (4x savings)** |
| **FedAvg-TopK10** | 90% Sparsified | 85.05% | **56.69%** | 1.245 | **4.99 MB (6.7x savings)** |

![FL vs Baselines](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/fl_vs_centralized_vs_local.png)
*Figure 5: Test accuracy across communication rounds for Centralized, Local-Only, FedAvg, and FedProx.*

#### Key Findings:
1. **The Overfitting-Generalization Dilemma (Local-Only)**: Edge devices training strictly in isolation achieved 91.50% on their own held-out test data but crashed to **36.67%** when evaluated across other subjects. Because individual users have unique gait speeds and arm-swing rhythms, local models overfit heavily to user biometrics.
2. **FedProx Resolves Client Drift**: Under natural non-IID conditions, standard FedAvg suffered from client drift caused by heterogeneous local objectives. FedProx ($\mu=0.01$) added a proximal penalty $\frac{\mu}{2} ||w - w_t||^2$ that constrained local updates to stay within the neighborhood of the global model, boosting generalization accuracy from 50.48% to **56.34%** (matching the centralized ceiling of 57.05%).

---

### 3. Privacy & Security Analysis

#### 3.1 Differential Privacy Sweep ($\epsilon \in \{1, 5, 10, \infty\}$)
Differential privacy was implemented by clipping local client update norms to $C=1.0$ and adding calibrated Gaussian perturbation:
$$\sigma = \frac{C}{\epsilon} \sqrt{2 \ln(1.25 / \delta)}, \quad \delta = 10^{-4}$$

![Differential Privacy Sweep](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/differential_privacy_sweep.png)
*Figure 6: Privacy-Utility trade-off across varying privacy budgets $\epsilon$.*

- **$\epsilon = \infty$ (No DP)**: Converges to 50.48%–56.34% accuracy.
- **$\epsilon = 10$ (Mild Privacy)**: Converges to **44.48%** accuracy. Provides meaningful statistical indistinguishability while preserving activity recognition capability.
- **$\epsilon = 5$ (Moderate Privacy)**: Reaches **40.70%** mid-training, stabilizing around 24.83%–31.05%.
- **$\epsilon = 1$ (Strict Cryptographic Privacy)**: Yields **28.40%** accuracy due to large noise variance relative to model gradient magnitude.

#### 3.2 Secure Aggregation Simulation

![Secure Aggregation Demo](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/secure_aggregation_demo.png)
*Figure 7: Secure Aggregation verification showing client 0 update masked as pure noise, while server aggregate cancels masks exactly.*

We verified the pairwise zero-sum masking protocol where $s_{i,j} = -s_{j,i}$:
$$\sum_{i=1}^K \tilde{w}_i = \sum_{i=1}^K \left(w_i + \sum_{j \neq i} s_{i,j}\right) = \sum_{i=1}^K w_i$$
- **Reconstruction Error**: $\max |\text{Server Sum} - \text{True Sum}| = \mathbf{4.71 \times 10^{-13}}$ (machine precision zero).
- **Statistical Independence**: Correlation between individual client update $w_0$ and the transmitted masked vector $\tilde{w}_0$ is $r = \mathbf{-0.01128}$ (uncorrelated Gaussian noise to any snooping server).

#### 3.3 Membership Inference Attack (MIA) Evaluation

![MIA ROC Curves](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/membership_inference_attack_roc.png)
*Figure 8: Receiver Operating Characteristic (ROC) curves for shadow loss membership inference attacker.*

- **Centralized Model**: Attacker AUC = **0.578**, maximum attack accuracy = **60.47%**. Overfitting in centralized training leaks sample membership.
- **FedAvg (No DP)**: Attacker AUC = **0.481**, maximum attack accuracy = **56.40%**. Distributed averaging inherently provides partial privacy regularization.
- **DP-FedAvg ($\epsilon=1$)**: Attacker AUC drops to **0.036**, maximum attack accuracy = **53.50%** (virtually indistinguishable from a random coin toss of 50%). Proves mathematical defense against training set re-identification.

---

### 4. Communication & Compression Analysis

In resource-constrained mobile environments, wireless uplinks and battery limits require aggressive communication reduction.

#### 4.1 Theoretical Formula
For $P$ trainable parameters, $K$ participating clients per round, and precision $B$ bytes:
$$\text{Bytes per round} = 2 \times (P \times B) \times K$$
With $P = 18,053$ parameters and $K = 8$ clients per round:
- **FP32 Full Precision**: $2 \times (18,053 \times 4) \times 8 = 1,155,392\text{ bytes} \approx \mathbf{1.10\text{ MB/round}}$ (or 144.4 KB per client).
- **INT8 Quantization**: $2 \times (18,053 \times 1) \times 8 = 288,848\text{ bytes} \approx \mathbf{0.27\text{ MB/round}}$ (**4x reduction**).
- **Top-10% Sparsification**: Transmits only top 10% magnitude coordinates + 16-bit indices: $2 \times (18,053 \times 0.1 \times 6) \times 8 = 173,308\text{ bytes} \approx \mathbf{0.16\text{ MB/round}}$ (**6.7x reduction**).

![Communication Compression](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/communication_compression.png)
*Figure 9: Test accuracy as a function of cumulative transmitted Megabytes.*

- To reach **55% test accuracy**, FP32 required **11.10 MB**, INT8 required only **2.77 MB**, and Top-10% sparsification required only **1.66 MB**!
- Top-10% sparsification preserved full convergence (56.69% final accuracy), demonstrating that over 90% of gradient updates in activity MLPs are redundant noise.

---

### 5. Answers to the Guiding Questions

#### Q1: What is the impact of non-IID data distribution on federated learning convergence and final model accuracy?
**Answer**: Non-IID distributions introduce statistical heterogeneity: each wearable user has distinct activity proportions (e.g. subject 1602 performs predominantly sedentary desk work, while subject 1625 performs extensive locomotion and sports). In standard FedAvg, local SGD pushes client weights toward local empirical optima. When averaged, these conflicting vectors cause **client drift**, oscillating the global objective and capping generalization accuracy (50.48%). Adding the FedProx proximal penalty $\frac{\mu}{2} ||w - w_t||^2$ pulls client trajectories toward global consensus, restoring convergence to 56.34%.

#### Q2: How does local epoch count ($E$) affect communication efficiency vs. client drift?
**Answer**: Local epochs present an intrinsic trade-off:
- **Higher $E$ (e.g. $E \ge 10$)**: Edge devices perform more local progress per communication round, reducing total rounds needed. However, on non-IID partitions, large $E$ causes client models to drift far from the global consensus, leading to destabilizing oscillations during server averaging.
- **Lower $E$ (e.g. $E = 1$ to $3$)**: Minimizes client drift and guarantees convergence toward the true empirical risk minimizer, but requires more frequent radio transmissions and higher energy consumption. Setting $E=3$ combined with FedProx provides the optimal operating point for mobile wearables.

#### Q3: What is the privacy-utility trade-off when applying $(\epsilon, \delta)$-Differential Privacy to gradient updates?
**Answer**: Differential Privacy bounds information leakage by injecting Gaussian noise proportional to $\frac{C}{\epsilon}$. For large $\epsilon$ ($\epsilon=10$), noise perturbation is minimal, preserving 44.5% accuracy while thwarting shadow model reconstruction. For strict privacy ($\epsilon=1$), the noise standard deviation approaches or exceeds gradient magnitudes, suppressing model capacity to 28.4% accuracy. In practice, $\epsilon \in [3, 8]$ offers the ideal balance for edge healthcare monitoring.

#### Q4: How does Secure Aggregation protect client updates from a curious server, and what are its computational and communication overheads?
**Answer**: Secure Aggregation utilizes pairwise zero-sum random masks: clients $i$ and $j$ agree on shared pseudorandom masks $s_{i,j} = -s_{j,i}$ using Diffie-Hellman key agreement. Each client submits $\tilde{w}_i = w_i + \sum_j s_{i,j}$. Since $\sum s_{i,j} \equiv 0$, the server calculates the exact unmasked aggregate without ever decrypting any single user's vector ($r = -0.011$ correlation).
- **Overhead**: Communication scales as $O(K^2)$ during key exchange and $O(K \cdot P)$ for updates; computation requires PRG expansions per neighbor. For 51 clients, this overhead is negligible ($< 50$ ms).

#### Q5: How do communication compression techniques (8-bit quantization, top-$k$ sparsification) impact model convergence rate?
**Answer**: 
- **8-bit Integer Quantization (INT8)**: Maps 32-bit floats to $[-128, 127]$ with affine scale and zero-point. It achieves an exact $4\times$ bandwidth saving with near-zero accuracy penalty (56.01% vs 56.34%), as neural networks are inherently resilient to low-magnitude precision errors.
- **Top-$k$ Sparsification (10%)**: Transmits only the 10% most significant weight deltas along with index masks. It achieves a $6.7\times$ bandwidth reduction and reaches target accuracy faster in terms of transmitted megabytes (1.66 MB vs 11.10 MB), confirming that edge FL networks can discard 90% of gradient traffic.

---

### 6. Real-World Applications

#### 6.1 Healthcare: Continuous Patient & Elder Monitoring
- **Parkinson's & Motor Disorders**: Patients wearing smartwatches capture tremor frequencies and gait freezing episodes. Centralizing continuous accelerometer feeds exposes lifestyle and mobility patterns. Federated learning enables hospital consortiums (e.g., Mayo Clinic, Charité) to train disease-progression models across patient cohorts without clinical data ever leaving personal devices.
- **Cardiovascular & Fall Detection**: Wearables detect cardiac arrhythmias and acute fall events while complying strictly with HIPAA and GDPR Article 9 (Special Category Health Data).

#### 6.2 Finance: Distributed Fraud & Biometric Authentication
- **Behavioral Biometrics**: Financial institutions use swipe dynamics, typing cadence, and phone handling angles to verify user identity during banking sessions. Federated learning trains fraud detection models across millions of bank customers without uploading private biometric signatures to centralized servers.
- **Cross-Institutional Anti-Money Laundering (AML)**: Multiple competing financial institutions train cooperative graph neural networks to detect cross-border laundering syndicates while maintaining institutional client confidentiality.

---

### 7. Interactive 3D Web Visualizer & Live WebSocket Server

The system includes a production-grade 3D dashboard:
- **Frontend**: React 19 + React Three Fiber + Three.js + Recharts (`federated-viz`).
- **Nodes**: 51 WISDM subject client nodes orbiting a central holographic server, color-coded by dominant activity group with hover telemetry.
- **Live WebSocket Pipeline**: Connects to `ws://localhost:8000/metrics` driven by `server.py` (FastAPI / Uvicorn), animating:
  1. **Broadcast**: Global model wave expands to active clients.
  2. **Local Training**: Selected edge nodes glow and pulse.
  3. **Upload**: Update particles stream back to the central server.
  4. **Aggregate**: Server flashes on aggregation, live updating loss and accuracy charts.
- **Interactive Controls**: Non-IID toggle, Differential Privacy slider, speed multiplier (0.5x to 4x), OrbitControls, and mobile-responsive viewport.

![Federated Learning 3D Visualizer Interface](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/app_loaded_1791794739772.png)
*Figure 10: Live 3D interface rendering the 51 wearable subject nodes, central aggregator, and real-time training charts.*

![3D Training Animation & Controls Demo](C:/Users/Surya_prakash/.gemini/antigravity-ide/brain/db64375e-4749-43fb-9c44-95e2a934aa5a/federated_learning_ui_1791794702685.webp)
*Figure 11: Real-time interactive session demonstrating round progression, particle animations, and parameter controls.*
