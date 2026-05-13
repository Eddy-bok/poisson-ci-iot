# Certified Real-Time Anomaly Detection for IoT Networks with Pre-Deployment Verification

**Authors:** S. Spektor and E. Ibokete  
**Affiliation:** School of Data, Computing & Mathematics, Canisius University, Buffalo, NY  
**Paper:** Submitted to IEEE Transactions on Dependable and Secure Computing (TDSC)

---

## Overview

This repository contains code and configuration files to reproduce all experimental results reported in the paper. The certified anomaly detector is built on two new concentration inequalities for weighted sums of independent Poisson random variables (Theorems 3 and 6 in the paper) and uses the normalized deviation from the certified interval as a continuous anomaly score.

**Key results reproduced by this package:**
- AUC = 0.9648 on CIC IoT-DIAD 2024 (primary dataset, 8 attack families)
- F1 = 0.9708 at matched FPR = 1%, vs 0.1554 for Isolation Forest and 0.1865 for One-Class SVM
- Cross-dataset validation on CICIoT2023 (105 devices, 33 attack types) without retraining

---

## Datasets

The datasets are publicly available but must be downloaded separately (total ~2 GB):

**CIC IoT-DIAD 2024 (primary and roc-analysis):**  
Download from: https://www.unb.ca/cic/datasets/iot-diad-2024.html  
Required files: `BenignTraffic1.csv`, `DictionaryBruteForce.csv`, `DDoS-TCP_Flood.csv`, `DDoS-ICMP_Fragmentation.csv`, `Mirai-greip_flood.csv`, `VulnerabilityScan.csv`, `SqlInjection.csv`, `DoS-UDP_Flood.csv`, `DoS-TCP_Flood.csv`

**CICIoT2023 (cross-dataset validation):**  
Download from: https://www.unb.ca/cic/datasets/iotdataset-2023.html  
Required files: `BenignTraffic.pcap.csv`, `DDoS-TCP_Flood.pcap.csv`, `DoS-UDP_Flood.pcap.csv`, `Recon-PortScan.pcap.csv`, `DictionaryBruteForce.pcap.csv`

Place the downloaded CSVs in a directory of your choice and update the path in the notebook configuration cell.

---

## Installation

```bash
git clone https://github.com/[GITHUB_USERNAME]/poisson-ci-iot.git
cd poisson-ci-iot
pip install -r requirements.txt
```

Python 3.9+ required. Tested on Google Colab (Python 3.10, Ubuntu 22.04).

---

## Reproducing Results

### Option A: Google Colab (recommended)

#### Primary CIC IoT-DIAD 2024 experiments

1. Upload `notebooks/CIC_IoT2024_IDAD_primary.ipynb` to Google Colab
2. Upload the required CIC IoT-DIAD 2024 CSVs to your Google Drive
3. Update the dataset path in the configuration cell
4. Run all cells (Runtime > Run all)
5. Expected runtime: 30–60 minutes on Colab GPU/CPU

#### ROC and matched-FPR analysis

1. Upload `notebooks/CIC_IoT2024_IDAD_roc_analysis.ipynb` to Google Colab
2. Upload the required CIC IoT-DIAD 2024 CSVs to your Google Drive
3. Update the dataset path in the configuration cell
4. Run all cells
5. Expected runtime: 15–30 minutes

#### Cross-dataset validation (CICIoT2023)

1. Upload `notebooks/CICIoT2023_crossdataset.ipynb` to Google Colab
2. Upload the required CICIoT2023 CSVs to your Google Drive
3. Update the dataset path in the configuration cell
4. Run all cells
5. Expected runtime: 30–60 minutes

### Option B: Local execution

From the repository root:

```bash
bash scripts/reproduce_all.sh


Requires Jupyter and nbconvert installed locally. Results are saved to `results/`.

```


## Repository Structure

```text
poisson-ci-iot/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── configs/
│   ├── CIC_IoT2024_IDAD_primary_config.json
│   ├── CIC_IoT2024_IDAD_roc_analysis_config.json
|   └── CICIoT2023_crossdataset_config.json
│
├── data/                             Local dataset directory (not tracked)
│
├── src/
│   ├── __init__.py
│   ├── poisson_ci_detector.py        Poisson-CI detector implementations
│   ├── baselines.py                  Baseline anomaly detection models
│   ├── features.py                   Feature engineering and preprocessing
│   └── evaluation.py                 ROC analysis and evaluation utilities
│
├── notebooks/
│   ├── CIC_IoT2024_IDAD_primary.ipynb
│   ├── CIC_IoT2024_IDAD_roc_analysis.ipynb
│   └── CICIoT2023_crossdataset.ipynb
│
├── scripts/
│   └── reproduce_all.sh              One-command reproduction script
│
└── results/                          Generated figures, tables, and executed notebooks
```

---

## Feature Set (15 features)

| # | Feature | Description |
|---|---------|-------------|
| 1 | event_count | Total packets in bin |
| 2 | TCP | TCP packet count |
| 3 | UDP | UDP packet count |
| 4 | ICMP | ICMP packet count |
| 5 | log_event | log(1 + event_count) |
| 6 | log_TCP | log(1 + TCP) |
| 7 | log_UDP | log(1 + UDP) |
| 8 | log_ICMP | log(1 + ICMP) |
| 9 | total_bytes | Sum of packet sizes in bin |
| 10 | mean_flow_duration | Mean flow duration in bin |
| 11 | SYN_count | TCP SYN flag count |
| 12 | IAT_mean | Mean inter-arrival time |
| 13 | IAT_std | Std dev of inter-arrival time |
| 14 | IAT_cv | Coefficient of variation of IAT |
| 15 | IAT_mad | Median absolute deviation of IAT |

---

## Configuration

Configuration files documenting the experimental settings
used for each dataset are provided in `configs/`.

| Parameter | CIC IoT-DIAD 2024 | CICIoT2023 |
|-----------|-------------------|-------------|
| Bin width | 10 ms | 5 ms |
| α (significance) | 0.05 | 0.05 |
| κ floor | 0.1 | 0.1 |
| κ ceiling | 2.5 | 2.5 |
| K neighbors | 50 | 50 |
| Random seed | 42 | 42 |
| Train/Val/Test | 60/20/20 | 60/20/20 |

Configuration summaries:
- `configs/CIC_IoT2024_IDAD_primary_config.json`
- `configs/CICIoT2023_crossdataset_config.json`
- `configs/CIC_IoT2024_IDAD_roc_analysis_config.json`

---

## Baselines

Seven baselines are compared under matched false-positive rates:

1. **Event 3-sigma** — z-score on total event count
2. **Layered 3-sigma** — max z-score across protocol channels
3. **Isolation Forest** — scikit-learn, 100 estimators, contamination=0.05
4. **One-Class SVM** — RBF kernel, ν=0.05
5. **LSTM Autoencoder** — grid search over {32,64,128} hidden, {5,10,20} window
6. **Transformer Autoencoder** — same grid search
7. **Dispersion-MAD-Z** — VMR with MAD-normalized z-score threshold

---

## Expected Output

Running `CIC_IoT2024_IDAD_roc_analysis.ipynb` should produce:

```
ROC ANALYSIS — CIC IoT-DIAD 2024 (PRIMARY DATASET), 10ms
Method                    AUC   F1@0.001   F1@0.005   F1@0.01   F1@0.02   F1@0.05
Poisson-CI             0.9648    0.9643     0.9684    0.9708    0.9726    0.9730
IsoForest              0.5080    0.1283     0.1474    0.1554    0.1721    0.1970
OC-SVM                 0.7172    0.1568     0.1722    0.1865    0.2196    0.3200
```

Numbers should match within rounding error (±0.001) due to random seed 42.

---

## Citation

If you use this code, please cite:

```bibtex
@article{spektor2026certified,
  title={Certified Real-Time Anomaly Detection for IoT Networks 
         with Pre-Deployment Verification},
  author={Spektor, S. and Ibokete, E.},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2026},
  note={Submitted}
}
```

---

## License

MIT License. See `LICENSE` for details.
