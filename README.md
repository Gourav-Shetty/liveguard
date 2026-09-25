# LiveGuard-EHMS: Edge-AI Arrhythmia Detection & Federated Learning System

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/edge--hardware-Raspberry%20Pi%203B%2B%20%2F%204-red.svg)](https://www.raspberrypi.com/)
[![Inference Engine](https://img.shields.io/badge/inference-Pure--NumPy%20(Zero--PyTorch%20on%20Pi)-brightgreen.svg)]()
[![Federated Learning](https://img.shields.io/badge/federated--learning-Flower%20(FedAvg)-orange.svg)](https://flower.ai/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

**LiveGuard-EHMS** (Edge Health Monitoring System) is an edge-native, real-time cardiac arrhythmia detection and privacy-preserving Federated Learning platform. 

It deploys high-accuracy deep learning onto resource-constrained edge hardware (**Raspberry Pi 3 Model B+**, 1 GB RAM) to monitor patients at the bedside without cloud dependence, while leveraging **Federated Learning (Flower FedAvg/FedProx)** across distributed patient partitions to train and improve global models without compromising patient privacy.

---

## 🌟 Key Highlights

* **Zero-Framework Edge Inference**: Unlike traditional edge deployments that choke low-RAM hardware with heavy deep learning runtimes (e.g. PyTorch or TensorFlow), LiveGuard uses an optimized **pure-NumPy 1D-CNN inference engine**.
  * **Inference Latency**: **~1.5 ms** per heartbeat on a 1.4 GHz ARM Cortex-A53.
  * **RAM Footprint**: **~120 MB total system memory** (leaving >850 MB free on a 1 GB Pi 3B+).
  * **Model Size**: Compressed to just **13.8 KB** (`stage1_weights.npz`).
* **Real-Time Streaming DSP**: Sample-by-sample 0.5–45 Hz Butterworth IIR bandpass filtering, notch filtering, Pan-Tompkins adaptive QRS detection, dynamic thresholding, and continuous Heart Rate (BPM) calculation.
* **Modular Multi-Source Hardware Drivers**:
  * **Physical Hardware**: AD8232 ECG sensor via ADS1115 (I2C 16-bit ADC), MCP3008 (SPI 10-bit ADC), or Arduino Serial bridge.
  * **Clinical Replay**: Built-in streaming driver to replay real patient recordings from the MIT-BIH Arrhythmia Database at precise sampling rates (360 Hz).
* **Live HTML5 Telemetry & Oscilloscope**: A standalone, zero-dependency browser oscilloscope ([`frontend/test_viewer.html`](frontend/test_viewer.html)) connecting via asynchronous WebSockets (`ws://0.0.0.0:8765`) to stream raw ECG waveforms, filtered signals, instant R-peak badges, BPM metrics, and real-time arrhythmia alarms.
* **Privacy-Preserving Federated Learning**: 38-client non-IID patient simulation engine using Flower (`flwr`), evaluating localized training rounds, weight aggregation (FedAvg/FedProx), and cross-patient generalization.
* **Two-Stage Arrhythmia Classification**:
  * **Stage 1 (Binary Anomaly Gate)**: Rapid anomaly filtering (Normal vs. Arrhythmia) with calibrated 96%+ abnormal beat recall.
  * **Stage 2 (Multiclass Diagnostic Classifier)**: Extensible AAMI EC57 5-class categorization (Normal `N`, Supraventricular `S`, Ventricular `V`, Fusion `F`, Unknown `Q`).

---

## 📐 System Architecture

```
                                  LIVEGUARD PIPELINE
                                  
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                            DATA ACQUISITION                                  │
 │   AD8232 Sensor (Skin Leads)   ───>  ADS1115 / MCP3008 ADC (SPI/I2C)         │
 │   MIT-BIH Database Replay     ───>  Clinical Stream Driver (360 Hz)          │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │ Raw Voltage Sample (every 2.7 ms)
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                       REAL-TIME SIGNAL PROCESSING (DSP)                       │
 │   • Butterworth IIR Bandpass Filter (0.5 Hz - 45 Hz)                          │
 │   • Baseline Wander & 50/60 Hz Powerline Noise Rejection                      │
 │   • Pan-Tompkins QRS Derivative & Adaptive Window Integration                 │
 │   • Dynamic R-Peak Trigger & Continuous Heart Rate (BPM) Estimation           │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │ 180-Sample Beat Window [-72 to +108]
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                      STAGE 1: EDGE INFERENCE (Pure-NumPy)                     │
 │   • Conv1D (16 ch, k=7) + Fused BN + ReLU + MaxPool(2)                        │
 │   • Conv1D (32 ch, k=5) + Fused BN + ReLU + AdaptiveAvgPool1D                 │
 │   • Dense FC (32 -> 16 -> 2) -> Softmax Classification                        │
 │   • Execution: ~1.5 ms per beat | Memory: ~720 bytes buffer                   │
 └───────────────────────┬───────────────────────────────┬───────────────────────┘
                         │                               │
       [Normal Rhythm]   │                               │ [Arrhythmia Detected]
                         ▼                               ▼
       ┌───────────────────────────┐   ┌─────────────────────────────────────────┐
       │   Local Health Telemetry  │   │      CRITICAL EMERGENCY ALERT           │
       │   • Continuous BPM Stream │   │   • Instant Visual Alarm Badge          │
       │   • Green Status Badges   │   │   • High Abnormal Probability (0.99+)   │
       └─────────────┬─────────────┘   └────────────────────┬────────────────────┘
                     │                                      │
                     └───────────────────┬──────────────────┘
                                         │
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                     TELEMETRY & VISUALIZATION SERVER                          │
 │   • Asynchronous WebSocket Server (port 8765)                                 │
 │   • HTML5 Canvas Oscilloscope Dashboard (frontend/test_viewer.html)  │
 │   • Real-Time Synchronized Waveform & Anomaly Badges                          │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │ Local Ring Buffer of Detected Beats
                                         ▼
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                      FEDERATED LEARNING (Flower FedAvg)                       │
 │   • Local Head Fine-Tuning: On-device gradient updates on local beats         │
 │   • Central Aggregator: Multi-client FedAvg weight averaging (Flower FL)      │
 │   • Zero Raw Data Transmission: Only encrypted/raw weight updates (Δw)        │
 └───────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Hardware Specifications & Pinouts

### Required Components
1. **Raspberry Pi 3 Model B+** (or Pi 4 / Pi Zero 2 W) running **Raspberry Pi OS Lite 64-bit**.
2. **AD8232 Single-Lead Heart Rate Monitor**.
3. **ADS1115 16-Bit I2C ADC** (or MCP3008 10-Bit SPI ADC).
4. **3-Lead Biomedical Electrode Cable & Disposable Gel Pads** (RA, LA, RL).

### Wiring Diagram (AD8232 + ADS1115 to Raspberry Pi)

| AD8232 Pin | ADS1115 Pin | Raspberry Pi 3B+ Pin | Description |
| :--- | :--- | :--- | :--- |
| **3.3V** | **VDD** | Pin 1 (3.3V Power) | Power Supply |
| **GND** | **GND** | Pin 6 (Ground) | Common Ground |
| **OUTPUT** | **A0** | — | Analog ECG Signal |
| **LO+** | — | Pin 16 (GPIO 23) | Leads-off Positive Detect |
| **LO-** | — | Pin 18 (GPIO 24) | Leads-off Negative Detect |
| — | **SDA** | Pin 3 (GPIO 2 / I2C SDA) | I2C Data |
| — | **SCL** | Pin 5 (GPIO 3 / I2C SCL) | I2C Clock |

---

## ⚡ Quick Start: Running on Raspberry Pi

### 1. Pi Setup (Ultra-Lightweight)
No PyTorch or heavy dependencies are needed on the Pi. Only install the standard Python math/signal packages:
```bash
sudo apt update
sudo apt install -y python3-numpy python3-scipy python3-websockets git
```

Clone the repository to your Raspberry Pi:
```bash
git clone https://github.com/Gourav-Shetty/liveguard.git ~/LiveGuard
cd ~/LiveGuard
```

### 2. Run with MIT-BIH Clinical Patient Replay
Test the complete real-time pipeline using recorded hospital patients:

* **Healthy Patient (#100)**:
  ```bash
  python3 -m backend.run_edge --source CLINICAL --patient 100
  ```
  *(Expected: Clean rhythm, steady 75-80 BPM, 0 false alarms).*

* **Arrhythmia Patient (#207 - Ventricular Bigeminy / PVC)**:
  ```bash
  python3 -m backend.run_edge --source CLINICAL --patient 207
  ```
  *(Expected: Live alerts flagging premature ventricular contractions with 0.99–1.00 probability).*

### 3. Run with Physical Sensors
```bash
# Using ADS1115 I2C ADC:
python3 -m backend.run_edge --source ADS1115

# Using MCP3008 SPI ADC:
python3 -m backend.run_edge --source RPI_SPI

# Using Arduino USB Bridge:
python3 -m backend.run_edge --source ARDUINO --port /dev/ttyUSB0
```

---

## 💻 Live Web Visualizer (HTML5 Oscilloscope)

LiveGuard includes an in-browser live oscilloscope in [`frontend/test_viewer.html`](frontend/test_viewer.html).

1. Ensure `run_edge.py` is running on the Pi (or laptop).
2. Open [`frontend/test_viewer.html`](frontend/test_viewer.html) in any modern browser (Chrome, Edge, Firefox, Safari).
3. Set the WebSocket URL:
   * Local: `ws://localhost:8765`
   * Over Local Network / Wi-Fi: `ws://raspberrypi.local:8765` (or replace with the Pi's IP address).
4. Click **Connect**:
   * **Sign in** (or **Create account** on your first run) — the live stream only starts after authentication.
   * Observe the live green ECG trace sweeping across the grid.
   * Watch the heart rate indicator update dynamically on every detected R-peak.
   * Inspect real-time status banners: green for **NORMAL BEAT**, flashing red for **ABNORMAL / ARRHYTHMIA**.

---

## 🔐 Authentication

The telemetry dashboard is protected by a username/password login (WebSocket handshake, no extra dependencies — Python stdlib `sqlite3`/`hashlib`/`hmac` only).

### Accounts
* **First user**: use *Create account* in the dashboard — registration is open only until the first account exists (bootstrap).
* **Later users**: an admin provisions them via the CLI (works regardless of the registration policy):
  ```bash
  python -m backend.auth.cli create-user <username>   # prompts for the password
  ```
* **Policy override** (env var): `LIVEGUARD_ALLOW_REGISTER=1` keeps registration open, `=0` keeps it closed (default: open only while zero users exist).

### How it works
* Passwords are stored as **PBKDF2-HMAC-SHA256** (100k iterations, per-user 16-byte salt) in `data/liveguard_users.db` (SQLite, gitignored) — never in plaintext.
* Logins return an **HMAC-signed token** (24 h TTL), signed with a server secret persisted in `data/.liveguard_secret` (gitignored — never commit it).
* Unauthenticated clients receive **no telemetry at all** until `auth_ok`.
* Rate limiting: 3 failed sign-ins per connection, 5 failed registrations per connection, and **10 failed sign-ins per client IP per 5 minutes** (the slot is reserved before the password is checked — parallel connection bursts can't exceed the cap — and refunded on success, so only failures count). Accounts are never locked by username: an attacker cannot deny service to a specific user, and the per-IP budget auto-expires after 5 minutes. Password hashing runs off the event loop, so login attempts can't stall live ECG delivery.
  * *Deployment caveat*: the budget is keyed by the TCP client IP. Devices behind the same NAT share one budget, and behind a TLS-terminating proxy every client would share the proxy's IP — in that case raise `MAX_IP_AUTH_FAILURES` in `backend/telemetry_server.py` or terminate TLS directly on the Python process.
* Logout is client-side (stateless tokens): a stolen token stays valid until it expires — rotate `data/.liveguard_secret` to revoke all sessions at once.

### Security assumptions
* Traffic runs over plain `ws://` — credentials are **cleartext on the network**. This is acceptable only on a trusted local/ lab network; put the server behind TLS (`wss://`) before exposing it beyond a LAN.

---

## 🧠 Model Training & Weight Export (Laptop / Workstation)

All neural network training, hyperparameter optimization, and threshold calibration are performed on the workstation/laptop using PyTorch.

### 1. Build Partitioned Datasets
Prepares non-IID patient splits from the MIT-BIH database:
```bash
python training/build_mitbih_splits.py
```

### 2. Centralized Model Training
Trains the Stage 1 1D-CNN on patient partitions with focal/weighted loss:
```bash
python training/quickstart-pytorch/train_centralized.py
```

### 3. Calibrate Anomaly Threshold
Computes optimal precision/recall thresholds for edge deployment:
```bash
python training/quickstart-pytorch/calibrate_stage1_threshold.py
```

### 4. Export Fused Weights to Pure-NumPy Format
Fuses all `BatchNorm1d` layers directly into `Conv1d` weights and outputs an ultra-compact `.npz` file for the Raspberry Pi:
```bash
python training/export_weights_to_npz.py
```
*(Produces `data/stage1_weights.npz` - 13.8 KB).*

---

## 🌐 Federated Learning with Flower (`flwr`)

LiveGuard simulates a federated hospital network where 38 distinct patient nodes collaborate to train an arrhythmia classification model without sharing raw ECG records.

### Architecture:
* **Server**: Coordinates federated rounds, samples clients, applies `FedAvg` or `FedProx` aggregation, and evaluates global performance on a held-out test set.
* **Clients**: Each client represents an isolated hospital partition (patient recording), training for local epochs before returning weight updates ($\Delta w$).

### Running the FL Simulation:
```bash
cd training/quickstart-pytorch
pip install -e .
flwr run . --stream
```

To configure hyperparameters (e.g. number of rounds, batch size, proximal $\mu$), edit [`training/quickstart-pytorch/pyproject.toml`](training/quickstart-pytorch/pyproject.toml):
```toml
[tool.flwr.app.config]
num-server-rounds = 20
local-epochs = 2
learning-rate = 0.001
batch-size = 32
proximal-mu = 0.0 # set > 0 for FedProx
```

---

## 📊 Benchmark & Performance Results

### Edge Performance on Raspberry Pi 3B+ (1.4 GHz Cortex-A53)
* **Single Beat Inference Time**: **1.42 ms**
* **End-to-End Latency** (Filter + Peak Detect + Inference): **~1.80 ms**
* **CPU Load at 360 Hz**: **< 4.5%**
* **Memory Footprint**: **~120 MB total system RAM**
* **Stage 1 Binary Detection Recall**: **96.9%** on MIT-BIH test arrhythmia beats.

---

## 📁 Repository Structure

```
LiveGuard/
├── backend/                           # Real-time edge execution pipeline (Pi-ready)
│   ├── arduino_bridge/                # Arduino firmware for ADC streaming
│   ├── auth/                          # Username/password auth (stdlib SQLite + WS handshake)
│   │   ├── cli.py                     # Admin CLI: create/list/delete users, change password
│   │   ├── db.py                      # SQLite user store (PBKDF2 password hashing)
│   │   └── service.py                 # Login/token service & registration policy
│   ├── drivers/                       # Hardware & simulation drivers
│   │   ├── ads1115_driver.py          # 16-bit I2C ADC driver
│   │   ├── clinical_driver.py         # Replay driver for MIT-BIH hospital records
│   │   ├── mock_driver.py             # Synthetic waveform generator for headless tests
│   │   ├── rpi_mcp3008_driver.py      # 10-bit SPI ADC driver
│   │   └── serial_driver.py           # USB Serial driver (Arduino / microcontrollers)
│   ├── config.py                      # Sampling rates, thresholds, and GPIO pin mapping
│   ├── edge_infer.py                  # Pure-NumPy 1D-CNN inference engine (<14 KB)
│   ├── requirements.txt               # Edge runtime dependencies
│   ├── run_edge.py                    # Main edge orchestrator
│   ├── signal_processing.py           # Butterworth IIR filter & Pan-Tompkins QRS detector
│   └── telemetry_server.py            # Asynchronous WebSocket broadcast server
│
├── frontend/                          # Browser dashboard (zero-dependency HTML5)
│   └── test_viewer.html               # Real-time HTML5 oscilloscope & alert visualizer
│
├── training/                          # Model training, calibration & weight export (workstation)
│   ├── quickstart-pytorch/            # Flower FL app & PyTorch training scripts
│   │   ├── pytorchexample/            # Flower client/server apps
│   │   │   ├── client_app.py          # Local client training logic
│   │   │   ├── server_app.py          # Global FedAvg aggregator & evaluator
│   │   │   └── task.py                # PyTorch 1D-CNN model & non-IID data loaders
│   │   ├── calibrate_stage1_threshold.py # Precision-recall threshold optimization
│   │   ├── plot_centralized_results.py   # Generates publication-ready ROC & PR curves
│   │   ├── pyproject.toml             # Flower application configuration
│   │   └── train_centralized.py       # Baseline centralized model training
│   ├── build_mitbih_splits.py         # Preprocesses raw MIT-BIH records into splits
│   ├── export_stage2_to_npz.py        # Exports Stage 2 multiclass weights
│   ├── export_weights_to_npz.py       # Fuses BatchNorm and exports Stage 1 weights
│   └── train_stage2_multiclass.py     # Stage 2 5-class AAMI arrhythmia classifier
│
├── docs/                              # Guides & documentation generators
│   ├── build_guide_docx.py            # Generates the hardware wiring guide
│   ├── build_laptop_guide_docx.py     # Generates the laptop + Arduino quickstart
│   ├── LiveGuard_Hardware_Guide.docx  # Hardware wiring & assembly documentation
│   └── LiveGuard_Laptop_Arduino_Quickstart.docx # Quickstart guide for laptop + Arduino setup
│
├── data/                              # Partitioned datasets, weights, and evaluation metrics
│   ├── fl_experiment_results.csv      # Centralized vs FL benchmark logs
│   ├── mitbih_test_ready.npz          # Held-out patient test set
│   ├── mitbih_train_ready.npz         # 38-patient non-IID training split
│   ├── mitbih_val_ready.npz           # Validation patient split
│   ├── stage1_cnn_centralized.pth     # Centralized PyTorch checkpoint
│   └── stage1_weights.npz             # Fused NumPy weights
│
├── archive/                           # Raw MIT-BIH waveform database (gitignored, not committed)
│
├── tests/                             # Regression suites (Python unittest + Node frontend harness)
│   ├── test_pipeline.py               # End-to-end pipeline & telemetry server tests
│   ├── test_auth_flow.py              # Auth, handshake & rate-limit tests
│   └── frontend_harness.js            # Headless viewer auth/broadcast tests
│
└── README.md                          # Master documentation
```

---

## 👥 Authors & Acknowledgments

* **Gourav Shetty** & **Mirza H** — *Final Year Major Project*
* **MIT-BIH Arrhythmia Database**: Mark, R., & Moody, G. (PhysioNet).
* **Flower FL Framework**: Adapted for decentralized healthcare simulations.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](training/quickstart-pytorch/LICENSE) file for details.
