# 🛠️ Edge Sensor Processing & Cloud Sync Pipeline
**Edge processing pipeline for industrial vibration sensors with AWS S3 sync (via LocalStack).**

---

## 📐 Project Architecture & File Structure

```text
edge-vibration-pipeline/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Global constants (sampling rates, S3 bucket name, path settings)
│   ├── sensor/
│   │   ├── __init__.py
│   │   └── simulator.py          # Part 1: RK4 Oscillator (signal generator & fault injection)
│   ├── processor/
│   │   ├── __init__.py
│   │   ├── worker.py             # Part 2: High-priority edge ingestion worker thread
│   │   ├── ring_buffer.py        # O(1) bounded-memory circular buffer (collections.deque)
│   │   ├── features.py           # Vectorized NumPy feature extraction (RMS, Mean, Std, FFT Peak)
│   │   └── detector.py           # Statistical AnomalyDetector tracking Precision & Recall
│   ├── sync/
│   │   ├── __init__.py
│   │   ├── worker.py             # Part 3: Background cloud sync worker (Exponential Backoff + Jitter)
│   │   ├── durable_queue.py      # Disk-backed persistent FIFO queue (.json.gz) with atomic writes
│   │   └── s3_uploader.py        # AWS S3 / LocalStack client with Hive partitioning & self-healing
│   └── utils/
│       ├── __init__.py
│       └── logger.py             # Edge-optimized rotating file logger
├── tests/
│   ├── test_simulator.py         # Tests for RK4 numerical integrator and fault window
│   ├── test_ring_buffer.py       # Tests for O(1) bounded memory ring buffer
│   ├── test_detector.py          # Tests for anomaly detection logic & confusion matrix
│   ├── test_durable_queue.py     # Tests for disk persistence & crash recovery
│   └── test_s3_uploader.py       # Tests for S3 uploads, Hive partitioning & self-healing
├── benchmark.py                  # Benchmark script for Throughput (>1000 samples/s) & Peak Memory
├── main.py                       # Main orchestrator (thread startup, signal handling, graceful shutdown)
├── docker-compose.yml            # LocalStack & application environment orchestration
└── requirements.txt              # Project dependencies

```

---

## 🚀 Execution Commands

### 1. Build and Start the Environment
```bash
# 1. Install Docker and adjust permissions.
sudo apt-get update && sudo apt-get install -y docker.io docker-compose
sudo usermod -aG docker $USER

# 2. Start the containers using 'sudo' in the first command to avoid session problems.
sudo docker-compose up -d

# 3. Wait for the package installation to finish (required the first time).
sudo docker exec -it edge-pipeline bash -c "while ! command -v pytest &> /dev/null; do sleep 1; done"
```
*(Note: If you encounter `permission denied`, run `docker-compose` and `docker` commands with `sudo`)*

---

### 2. Run All Automated Unit Tests

Run the full test suite covering the sensor simulator, bounded memory buffer, anomaly detector, durable queue, and S3 uploader:

```bash
sudo docker exec -it edge-pipeline pytest tests/ -v
```

---
### 3. Run Throughput & Memory Benchmark

Verify that the edge pipeline sustains **> 1000 samples/sec** on a single core and keeps a small, static **Peak Memory** footprint:

```bash
sudo docker exec -it edge-pipeline python benchmark.py
```

---
### 4. Check the docker and the S3 data

```bash
sudo docker ps
sudo docker exec -it edge-localstack awslocal s3 ls
sudo docker exec -it edge-localstack awslocal s3 ls s3://industrial-sensor-data --recursive
```

---
### 5. Run the Main Edge Application

Start the live sensor ingestion, rolling window feature extraction, anomaly detection, and cloud synchronization:

```bash
sudo docker exec -it edge-pipeline python main.py
```

> **Note:** Press `Ctrl + C` at any time to stop the application gracefully and print the final **Precision / Recall / F1-Score** report.

---

### 6. Inspect Uploaded Files on Mocked AWS S3 (LocalStack)

#### A. List all uploaded files (Hive-partitioned structure):

```bash
sudo docker exec -it edge-localstack awslocal s3 ls s3://industrial-sensor-data --recursive
```

#### B. Continuously watch files arriving in S3 (Live Watch Mode):

```bash
sudo docker exec -it edge-localstack watch -n 2 awslocal s3 ls s3://industrial-sensor-data --recursive
```

---

### 7. Demonstrate Network Failure & Self-Healing Resilience

1. While `main.py` is running in one terminal, **stop LocalStack** in a second terminal to simulate a network outage:

```bash
sudo docker stop edge-localstack
```

2. Observe `main.py` buffering payloads locally to disk.
3. Restart LocalStack to verify automatic connection recovery and queue drain:

```bash
sudo docker start edge-localstack
```

---

## 🛑 Teardown & Clean Up

To stop all containers and clear created volumes:

```bash
sudo docker-compose down -v
```