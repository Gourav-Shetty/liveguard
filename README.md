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
* **Live HTML5 Telemetry & Oscilloscope**: A standalone, zero-dependency browser oscilloscope ([`test_viewer.html`](test_viewer.html)) connecting via asynchronous WebSockets (`ws://0.0.0.0:8765`) to stream raw ECG waveforms, filtered signals, instant R-peak badges, BPM metrics, and real-time arrhythmia alarms.
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
 │   • HTML5 Canvas Oscilloscope Dashboard (test_viewer.html)                    │
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
  python3 -m edge_system.run_edge --source CLINICAL --patient 100
  ```
  *(Expected: Clean rhythm, steady 75-80 BPM, 0 false alarms).*

* **Arrhythmia Patient (#207 - Ventricular Bigeminy / PVC)**:
  ```bash
  python3 -m edge_system.run_edge --source CLINICAL --patient 207
  ```
  *(Expected: Live alerts flagging premature ventricular contractions with 0.99–1.00 probability).*

### 3. Run with Physical Sensors
```bash
# Using ADS1115 I2C ADC:
python3 -m edge_system.run_edge --source ADS1115

# Using MCP3008 SPI ADC:
python3 -m edge_system.run_edge --source RPI_SPI

# Using Arduino USB Bridge:
python3 -m edge_system.run_edge --source ARDUINO --port /dev/ttyUSB0
```

---

## 💻 Live Web Visualizer (HTML5 Oscilloscope)

LiveGuard includes an in-browser live oscilloscope in [`test_viewer.html`](test_viewer.html).

1. Ensure `run_edge.py` is running on the Pi (or laptop).
2. Open [`test_viewer.html`](test_viewer.html) in any modern browser (Chrome, Edge, Firefox, Safari).
3. Set the WebSocket URL:
   * Local: `ws://localhost:8765`
   * Over Local Network / Wi-Fi: `ws://raspberrypi.local:8765` (or replace with the Pi's IP address).
4. Click **Connect**:
   * Observe the live green ECG trace sweeping across the grid.
   * Watch the heart rate indicator update dynamically on every detected R-peak.
   * Inspect real-time status banners: green for **NORMAL BEAT**, flashing red for **ABNORMAL / ARRHYTHMIA**.

---

## 🧠 Model Training & Weight Export (Laptop / Workstation)

All neural network training, hyperparameter optimization, and threshold calibration are performed on the workstation/laptop using PyTorch.

### 1. Build Partitioned Datasets
Prepares non-IID patient splits from the MIT-BIH database:
```bash
python build_mitbih_splits.py
```

### 2. Centralized Model Training
Trains the Stage 1 1D-CNN on patient partitions with focal/weighted loss:
```bash
python quickstart-pytorch/train_centralized.py
```

### 3. Calibrate Anomaly Threshold
Computes optimal precision/recall thresholds for edge deployment:
```bash
python quickstart-pytorch/calibrate_stage1_threshold.py
```

### 4. Export Fused Weights to Pure-NumPy Format
Fuses all `BatchNorm1d` layers directly into `Conv1d` weights and outputs an ultra-compact `.npz` file for the Raspberry Pi:
```bash
python export_weights_to_npz.py
```
*(Produces `quickstart-pytorch/stage1_weights.npz` - 13.8 KB).*

---

## 🌐 Federated Learning with Flower (`flwr`)

LiveGuard simulates a federated hospital network where 38 distinct patient nodes collaborate to train an arrhythmia classification model without sharing raw ECG records.

### Architecture:
* **Server**: Coordinates federated rounds, samples clients, applies `FedAvg` or `FedProx` aggregation, and evaluates global performance on a held-out test set.
* **Clients**: Each client represents an isolated hospital partition (patient recording), training for local epochs before returning weight updates ($\Delta w$).

### Running the FL Simulation:
```bash
cd quickstart-pytorch
pip install -e .
flwr run . --stream
```

To configure hyperparameters (e.g. number of rounds, batch size, proximal $\mu$), edit [`quickstart-pytorch/pyproject.toml`](quickstart-pytorch/pyproject.toml):
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
├── edge_system/                     # Real-time edge execution pipeline (Pi-ready)
│   ├── arduino_bridge/              # Arduino firmware for ADC streaming
│   ├── drivers/                     # Hardware & simulation drivers
│   │   ├── ads1115_driver.py        # 16-bit I2C ADC driver
│   │   ├── clinical_driver.py       # Replay driver for MIT-BIH hospital records
│   │   ├── mock_driver.py           # Synthetic waveform generator for headless tests
│   │   ├── rpi_mcp3008_driver.py    # 10-bit SPI ADC driver
│   │   └── serial_driver.py         # USB Serial driver (Arduino / microcontrollers)
│   ├── config.py                    # Sampling rates, thresholds, and GPIO pin mapping
│   ├── edge_infer.py                # Pure-NumPy 1D-CNN inference engine (<14 KB)
│   ├── run_edge.py                  # Main edge orchestrator
│   ├── signal_processing.py         # Butterworth IIR filter & Pan-Tompkins QRS detector
│   └── telemetry_server.py          # Asynchronous WebSocket broadcast server
│
├── quickstart-pytorch/              # Federated Learning & PyTorch training module
│   ├── pytorchexample/              # Flower client/server apps
│   │   ├── client_app.py            # Local client training logic
│   │   ├── server_app.py            # Global FedAvg aggregator & evaluator
│   │   └── task.py                  # PyTorch 1D-CNN model & non-IID data loaders
│   ├── calibrate_stage1_threshold.py# Precision-recall threshold optimization
│   ├── plot_centralized_results.py  # Generates publication-ready ROC & PR curves
│   ├── pyproject.toml               # Flower application configuration
│   ├── stage1_weights.npz           # Exported NumPy weights for edge deployment
│   └── train_centralized.py         # Baseline centralized model training
│
├── data/                            # Partitioned datasets, weights, and evaluation metrics
│   ├── fl_experiment_results.csv    # Centralized vs FL benchmark logs
│   ├── mitbih_test_ready.npz        # Held-out patient test set
│   ├── mitbih_train_ready.npz       # 38-patient non-IID training split
│   ├── mitbih_val_ready.npz         # Validation patient split
│   ├── stage1_cnn_centralized.pth   # Centralized PyTorch checkpoint
│   └── stage1_weights.npz           # Fused NumPy weights
│
├── build_mitbih_splits.py           # Preprocesses raw MIT-BIH records into splits
├── export_weights_to_npz.py         # Fuses BatchNorm and exports Stage 1 weights
├── export_stage2_to_npz.py          # Exports Stage 2 multiclass weights
├── train_stage2_multiclass.py       # Stage 2 5-class AAMI arrhythmia classifier
├── test_viewer.html                 # Real-time HTML5 oscilloscope & alert visualizer
├── LiveGuard_Hardware_Guide.docx    # Hardware wiring & assembly documentation
├── LiveGuard_Laptop_Arduino_Quickstart.docx # Quickstart guide for laptop + Arduino setup
└── README.md                        # Master documentation
```

---

## 👥 Authors & Acknowledgments

* **Gourav Shetty** & **Mirza H** — *Final Year Major Project*
* **MIT-BIH Arrhythmia Database**: Mark, R., & Moody, G. (PhysioNet).
* **Flower FL Framework**: Adapted for decentralized healthcare simulations.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](quickstart-pytorch/LICENSE) file for details.
