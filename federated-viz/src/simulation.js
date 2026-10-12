/**
 * FedAvg Simulation Engine
 * ========================
 * Runs genuine federated averaging on synthetic activity-recognition data
 * entirely in the browser. Produces real accuracy/loss curves.
 */

// ─── Synthetic Data Generation ──────────────────────────────
function generateSyntheticActivityData(numClients = 8, samplesPerClient = 200, numFeatures = 20, numClasses = 6, nonIID = false) {
  const clients = [];
  // Class centers — each class has a distinct centroid in feature space
  const classCenters = [];
  for (let c = 0; c < numClasses; c++) {
    const center = [];
    for (let f = 0; f < numFeatures; f++) {
      center.push((Math.random() - 0.5) * 4 + c * 1.5);
    }
    classCenters.push(center);
  }

  for (let i = 0; i < numClients; i++) {
    const X = [];
    const y = [];

    if (nonIID) {
      // Non-IID: each client primarily has 2 dominant classes
      const dominantClasses = [i % numClasses, (i + 1) % numClasses];
      for (let s = 0; s < samplesPerClient; s++) {
        const classIdx = Math.random() < 0.8
          ? dominantClasses[Math.floor(Math.random() * 2)]
          : Math.floor(Math.random() * numClasses);
        const sample = classCenters[classIdx].map(v => v + (Math.random() - 0.5) * 2);
        X.push(sample);
        y.push(classIdx);
      }
    } else {
      // IID: uniform class distribution
      for (let s = 0; s < samplesPerClient; s++) {
        const classIdx = Math.floor(Math.random() * numClasses);
        const sample = classCenters[classIdx].map(v => v + (Math.random() - 0.5) * 2);
        X.push(sample);
        y.push(classIdx);
      }
    }
    clients.push({ X, y, id: i });
  }

  // Global test set
  const testX = [];
  const testY = [];
  for (let s = 0; s < 500; s++) {
    const classIdx = Math.floor(Math.random() * numClasses);
    const sample = classCenters[classIdx].map(v => v + (Math.random() - 0.5) * 2);
    testX.push(sample);
    testY.push(classIdx);
  }

  return { clients, testX, testY, numFeatures, numClasses, classCenters };
}

// ─── Simple Neural Network (single hidden layer) ───────────
class SimpleNet {
  constructor(inputSize, hiddenSize, outputSize) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.W1 = this.randomMatrix(inputSize, hiddenSize, 0.1);
    this.b1 = new Float32Array(hiddenSize);
    this.W2 = this.randomMatrix(hiddenSize, outputSize, 0.1);
    this.b2 = new Float32Array(outputSize);
  }

  randomMatrix(rows, cols, scale) {
    const m = new Float32Array(rows * cols);
    for (let i = 0; i < m.length; i++) {
      m[i] = (Math.random() - 0.5) * 2 * scale;
    }
    return m;
  }

  forward(x) {
    // Hidden layer: ReLU(x @ W1 + b1)
    const hidden = new Float32Array(this.hiddenSize);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += x[i] * this.W1[i * this.hiddenSize + j];
      }
      hidden[j] = Math.max(0, sum); // ReLU
    }

    // Output layer: softmax(hidden @ W2 + b2)
    const logits = new Float32Array(this.outputSize);
    let maxLogit = -Infinity;
    for (let j = 0; j < this.outputSize; j++) {
      let sum = this.b2[j];
      for (let i = 0; i < this.hiddenSize; i++) {
        sum += hidden[i] * this.W2[i * this.outputSize + j];
      }
      logits[j] = sum;
      if (sum > maxLogit) maxLogit = sum;
    }

    // Stable softmax
    const probs = new Float32Array(this.outputSize);
    let sumExp = 0;
    for (let j = 0; j < this.outputSize; j++) {
      probs[j] = Math.exp(logits[j] - maxLogit);
      sumExp += probs[j];
    }
    for (let j = 0; j < this.outputSize; j++) {
      probs[j] /= sumExp;
    }

    return { hidden, logits, probs };
  }

  getParams() {
    return {
      W1: new Float32Array(this.W1),
      b1: new Float32Array(this.b1),
      W2: new Float32Array(this.W2),
      b2: new Float32Array(this.b2),
    };
  }

  setParams(params) {
    this.W1.set(params.W1);
    this.b1.set(params.b1);
    this.W2.set(params.W2);
    this.b2.set(params.b2);
  }

  // SGD training step
  trainStep(X, y, lr = 0.01) {
    let totalLoss = 0;

    for (let s = 0; s < X.length; s++) {
      const { hidden, probs } = this.forward(X[s]);
      const target = y[s];

      // Cross-entropy loss
      totalLoss -= Math.log(Math.max(probs[target], 1e-10));

      // Gradients — output layer
      const dLogits = new Float32Array(probs);
      dLogits[target] -= 1;

      // dW2, db2
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.outputSize; j++) {
          this.W2[i * this.outputSize + j] -= lr * hidden[i] * dLogits[j];
        }
      }
      for (let j = 0; j < this.outputSize; j++) {
        this.b2[j] -= lr * dLogits[j];
      }

      // dHidden
      const dHidden = new Float32Array(this.hiddenSize);
      for (let i = 0; i < this.hiddenSize; i++) {
        let grad = 0;
        for (let j = 0; j < this.outputSize; j++) {
          grad += this.W2[i * this.outputSize + j] * dLogits[j];
        }
        dHidden[i] = hidden[i] > 0 ? grad : 0; // ReLU derivative
      }

      // dW1, db1
      for (let i = 0; i < this.inputSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          this.W1[i * this.hiddenSize + j] -= lr * X[s][i] * dHidden[j];
        }
      }
      for (let j = 0; j < this.hiddenSize; j++) {
        this.b1[j] -= lr * dHidden[j];
      }
    }

    return totalLoss / X.length;
  }

  evaluate(X, y) {
    let correct = 0;
    let totalLoss = 0;

    for (let s = 0; s < X.length; s++) {
      const { probs } = this.forward(X[s]);
      let maxIdx = 0;
      for (let j = 1; j < this.outputSize; j++) {
        if (probs[j] > probs[maxIdx]) maxIdx = j;
      }
      if (maxIdx === y[s]) correct++;
      totalLoss -= Math.log(Math.max(probs[y[s]], 1e-10));
    }

    return {
      accuracy: correct / X.length,
      loss: totalLoss / X.length,
    };
  }
}

// ─── FedAvg Algorithm ────────────────────────────────────────
function addNoise(params, epsilon, sensitivity = 1.0) {
  if (epsilon <= 0) return params;

  const sigma = sensitivity / epsilon;
  const noisy = {};
  for (const key of Object.keys(params)) {
    noisy[key] = new Float32Array(params[key].length);
    for (let i = 0; i < params[key].length; i++) {
      // Box-Muller for Gaussian noise
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      noisy[key][i] = params[key][i] + sigma * z;
    }
  }
  return noisy;
}

function clipUpdate(update, globalParams, maxNorm) {
  const clipped = {};
  let totalNorm = 0;

  for (const key of Object.keys(update)) {
    for (let i = 0; i < update[key].length; i++) {
      const diff = update[key][i] - globalParams[key][i];
      totalNorm += diff * diff;
    }
  }
  totalNorm = Math.sqrt(totalNorm);

  const scale = Math.min(1, maxNorm / (totalNorm + 1e-10));

  for (const key of Object.keys(update)) {
    clipped[key] = new Float32Array(update[key].length);
    for (let i = 0; i < update[key].length; i++) {
      clipped[key][i] = globalParams[key][i] + (update[key][i] - globalParams[key][i]) * scale;
    }
  }

  return clipped;
}

function averageParams(paramsList) {
  const avg = {};
  const n = paramsList.length;
  const keys = Object.keys(paramsList[0]);

  for (const key of keys) {
    avg[key] = new Float32Array(paramsList[0][key].length);
    for (let i = 0; i < avg[key].length; i++) {
      let sum = 0;
      for (let c = 0; c < n; c++) {
        sum += paramsList[c][key][i];
      }
      avg[key][i] = sum / n;
    }
  }

  return avg;
}

// ─── Main Simulation Runner ──────────────────────────────────
export class FedAvgSimulation {
  constructor(config = {}) {
    this.numClients = config.numClients || 8;
    this.numRounds = config.numRounds || 50;
    this.localEpochs = config.localEpochs || 5;
    this.clientsPerRound = config.clientsPerRound || 8;
    this.lr = config.lr || 0.01;
    this.nonIID = config.nonIID || false;
    this.dpEpsilon = config.dpEpsilon || 0; // 0 = no DP
    this.clipNorm = config.clipNorm || 5.0;

    this.numFeatures = 20;
    this.hiddenSize = 32;
    this.numClasses = 6;

    this.data = null;
    this.globalModel = null;
    this.centralizedModel = null;

    this.history = {
      federated: [],
      centralized: [],
    };

    this.currentRound = 0;
    this.isRunning = false;
    this.phase = 'idle'; // idle, broadcast, training, upload, aggregate
    this.activeClients = [];
  }

  initialize() {
    this.data = generateSyntheticActivityData(
      this.numClients, 200, this.numFeatures, this.numClasses, this.nonIID
    );

    this.globalModel = new SimpleNet(this.numFeatures, this.hiddenSize, this.numClasses);
    this.centralizedModel = new SimpleNet(this.numFeatures, this.hiddenSize, this.numClasses);
    this.centralizedModel.setParams(this.globalModel.getParams());

    this.currentRound = 0;
    this.history = { federated: [], centralized: [] };
  }

  computeBytes() {
    const totalParams = this.numFeatures * this.hiddenSize + this.hiddenSize +
      this.hiddenSize * this.numClasses + this.numClasses;
    // 4 bytes per float32, upload + download, per participating client
    return totalParams * 4 * this.clientsPerRound * 2;
  }

  async runRound(onPhaseChange) {
    if (this.currentRound >= this.numRounds) return null;

    const globalParams = this.globalModel.getParams();

    // Phase 1: Broadcast
    if (onPhaseChange) onPhaseChange('broadcast');
    await this.sleep(600);

    // Phase 2: Select clients and train locally
    const selectedClients = this.selectClients();
    this.activeClients = selectedClients.map(c => c.id);
    if (onPhaseChange) onPhaseChange('training', this.activeClients);
    await this.sleep(800);

    const clientUpdates = [];
    for (const client of selectedClients) {
      const localModel = new SimpleNet(this.numFeatures, this.hiddenSize, this.numClasses);
      localModel.setParams(globalParams);

      for (let e = 0; e < this.localEpochs; e++) {
        localModel.trainStep(client.X, client.y, this.lr);
      }

      let update = localModel.getParams();

      // DP: clip and noise
      if (this.dpEpsilon > 0) {
        update = clipUpdate(update, globalParams, this.clipNorm);
        update = addNoise(update, this.dpEpsilon);
      }

      clientUpdates.push(update);
    }

    // Phase 3: Upload updates
    if (onPhaseChange) onPhaseChange('upload', this.activeClients);
    await this.sleep(600);

    // Phase 4: Aggregate
    if (onPhaseChange) onPhaseChange('aggregate');
    await this.sleep(400);

    const avgParams = averageParams(clientUpdates);
    this.globalModel.setParams(avgParams);

    // Evaluate federated model
    const fedEval = this.globalModel.evaluate(this.data.testX, this.data.testY);

    // Centralized step (train on all data pooled)
    const allX = this.data.clients.flatMap(c => c.X);
    const allY = this.data.clients.flatMap(c => c.y);
    // One epoch on all data
    this.centralizedModel.trainStep(allX, allY, this.lr * 0.5);
    const centEval = this.centralizedModel.evaluate(this.data.testX, this.data.testY);

    this.currentRound++;

    const roundResult = {
      round: this.currentRound,
      fedAccuracy: fedEval.accuracy,
      fedLoss: fedEval.loss,
      centAccuracy: centEval.accuracy,
      centLoss: centEval.loss,
      epsilon: this.dpEpsilon,
      bytes: this.computeBytes(),
      activeClients: this.activeClients,
    };

    this.history.federated.push({ round: this.currentRound, accuracy: fedEval.accuracy, loss: fedEval.loss });
    this.history.centralized.push({ round: this.currentRound, accuracy: centEval.accuracy, loss: centEval.loss });

    if (onPhaseChange) onPhaseChange('idle');

    return roundResult;
  }

  selectClients() {
    const shuffled = [...this.data.clients].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(this.clientsPerRound, this.numClients));
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  reset(config = {}) {
    Object.assign(this, config);
    this.initialize();
  }
}
