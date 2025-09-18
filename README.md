## 🚀 MCST: Maneuver Compensation Strong Tracker

MCST is an adaptive deep learning-based radar target tracking algorithm designed to handle **high-speed** and **highly maneuverable targets**. It leverages a **Bi-LSTM architecture** with a **dual-level attention mechanism** and a custom **Maneuver Compensation Unit (MCU)** to achieve robust and accurate tracking performance, even under extreme target dynamics and observation uncertainties.

> 📄 Accepted at *IEEE Transactions on Aerospace and Electronic Systems*  
> 📚 [Read the paper](https://doi.org/10.1109/TAES.2024.3484393)

---

## ✨ Features

- 🚀 Supports tracking of hypersonic, highly maneuverable targets
- 🧠 Built upon a Bi-LSTM architecture with:
  - **Predictor & Updater modules** inspired by Kalman filtering
  - **Dual-level attention module** for temporal and sample weighting
  - **Maneuver Compensation Unit (MCU)** using FFT-based residual analysis
- 🔄 Outputs both **state estimates** and **uncertainty covariance matrices**
- 📈 Trained and tested on a **custom 10,000-trajectory dataset**
- 📦 Modular and extensible codebase

---

## 🖼️ Model Architecture

![MCST Architecture](./MC-LSTM_architecturev4.png)  
*A hybrid Bi-LSTM + attention framework with maneuver compensation and uncertainty modeling.*

---

## 📁 Project Structure

```bash
MCST/
├── config.py            # Model and training configs
├── data/                # Dataset and preprocessing scripts
├── models/              # Model definitions (Predictor, MCU, Updater)
├── utils/               # Utilities (normalization, trajectory initialization)
├── train/               # Training pipeline
├── evaluation/          # Evaluation and metrics
├── log/                 # log
├── main.py
```

---

## 🧪 Quick Start

### 1. Setup Environment

```bash
conda create -n mcst python=3.9
conda activate mcst
```

### 2. Download Dataset

You can download from https://github.com/Shen-kl/OneManeuveringTarget3D.

### 3. Train the Model

```bash
python main.py 
```

### 4. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/mcst_best.pth
```

---

## 📊 Dataset Overview

MCST is trained on a synthetic dataset of **10,000 trajectories**, simulating targets with:

- Velocities up to **Mach 5**
- **Acceleration range**: 3g–7g
- Motion models: CV, CA, HCT, FCT
- Sampling interval: 0.4 seconds
- Noise added in **spherical coordinates**, then converted to Cartesian

Each trajectory contains **100 frames**, with **3 model transitions** per trajectory.

---

## 📈 Performance

MCST outperforms several SOTA model-based tracking algorithms, including:

- Single model: **FSTCKF** ,and **RNSTF** 
- Multiple models: **HGMM**, and **RIMM**

Especially in **miss detection** and **rapid maneuver** scenarios.

---

## 📜 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{shen2024mcst,
  title={MCST: An adaptive tracking algorithm for high-speed and highly maneuverable targets based on bidirectional LSTM network},
  author={Shen, Kailun and Yuan, Weiming and Yan, Junkun and Ma, Keke},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2024},
  publisher={IEEE}
}
```

---



## 📜 Changelog

#### 2025/9/18

#### Added

- We have added the functionality to export the model in **TorchScript** format.

#### Changed

- The computation of the location-dependent min–max bounds in min–max normalization has been revised.
  Specifically, the original scheme that relied on a fixed maximum velocity has been replaced by a hybrid strategy that blends the fixed maximum velocity with the model-estimated filtered velocity.

#### Fixed

- Fixed the lack of normalization for the gain matrix $W$ in the Update class. This yields more balanced gradient distributions across all model layers.

