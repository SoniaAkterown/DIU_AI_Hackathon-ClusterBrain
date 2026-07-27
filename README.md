<p align="center">
  <img src="https://github.com/SoniaAkterown/DIU_AI_Hackathon-ClusterBrain/blob/main/photo_2026-07-28_01-12-59.jpg" alt="ClusterBrain Banner" width="50%">
</p>

<h1 align="center">ClusterBrain 🧠</h1>

<h3 align="center">An Intelligent, Machine-Learning-Driven GPU Cluster Management System (DIU AI Hackathon)</h3>

<p align="center">
  <img src="https://img.shields.io/github/stars/SoniaAkterown/ClusterBrain?style=flat&logo=github&color=yellow" alt="Stars">
  <img src="https://img.shields.io/github/watchers/SoniaAkterown/ClusterBrain?style=flat&logo=github&label=views&color=blue" alt="Views">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.50.1-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PyTorch-2.12.1-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/XGBoost-2.2.0-2196F3?style=flat" alt="XGBoost">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.7.0-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-2.3.2-150458?style=flat&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-2.3.2-013243?style=flat&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Uvicorn-0.49.0-499848?style=flat" alt="Uvicorn">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
</p>
# 🧠 ClusterBrain

**ClusterBrain** is an intelligent, machine-learning-driven GPU Cluster Management System. Built for the **DIU AI Hackathon**, this project aims to predict hardware failures, intelligently schedule workloads to minimize resource waste, and forecast cluster operational costs using a robust 3-layer architecture comprising 9 specialized ML models.

---

## 🏗️ Architecture

ClusterBrain is divided into three distinct layers, each handling a specific domain of cluster management:

### 1. 🩺 The Doctor (Health & Maintenance)
Monitors real-time node telemetry to detect failures and predict hardware lifespan.
* **Anomaly Detector:** PyTorch Autoencoder processing live node metrics.
* **Remaining Useful Life (RUL) Predictor:** XGBoost Regressor estimating hours until failure.
* **Root Cause Classifier:** Random Forest Classifier identifying the specific hardware issue (e.g., Thermal Throttle, Memory Leak).

### 2. 📅 The Planner (Job Scheduling & Resource Optimization)
Analyzes incoming job requests to optimally place them across the cluster.
* **Job Runtime Predictor:** XGBoost Regressor forecasting job duration.
* **Resource Waste Predictor:** XGBoost Regressor estimating how much requested VRAM/CPU will actually be wasted.
* **Placement Ranker:** XGBoost model that intelligently ranks nodes based on health, capacity, and job requirements to ensure optimal load balancing.

### 3. 💰 The Accountant (Cost & User Analytics)
Tracks user behavior and predicts financial impact.
* **Session State Classifier:** Random Forest identifying active vs. abandoned user sessions.
* **User Behavior Clustering:** Scikit-Learn K-Means model segmenting users (e.g., Power Users vs. Wasteful Hoarders).
* **Monthly Cost Forecaster:** XGBoost Regressor predicting future operational costs based on historical lag and rolling mean features.

---

## 🛠️ Technology Stack & Versions

This project is built purely in Python, utilizing modern data science and web frameworks.

* **Python:** 3.10+
* **FastAPI:** Backend API serving the model inference.
* **Streamlit:** Interactive frontend dashboard (`1.50.1`)
* **PyTorch:** Autoencoder implementation (`2.12.1`)
* **XGBoost:** Gradient boosted tree regressors and classifiers (`2.2.0`)
* **Scikit-Learn:** Random forests, clustering, and data scaling (`1.7.0`)
* **Pandas / NumPy:** Data manipulation and feature engineering (`2.3.2`)
* **Uvicorn:** ASGI server for FastAPI (`0.49.0`)

---

## 🚀 How to Run Locally

### Using Docker (Recommended)
ClusterBrain is fully containerized for seamless deployment.

1. Ensure **Docker Desktop** is installed and running on your machine.
2. Open a terminal in the project directory and run:
```bash
docker-compose up -d --build
```
3. The FastAPI backend will be available at `http://127.0.0.1:8000`.
4. The Streamlit dashboard will be available at `http://localhost:8501`. 

### Manual Setup (Without Docker)
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
2. Start the FastAPI backend:
```bash
uvicorn app:app --reload
```
3. In a second terminal, launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

---

## 📁 Directory Structure
* `app.py`: FastAPI backend that dynamically loads models and serves inference endpoints.
* `dashboard.py`: Streamlit frontend with 6 interactive panels and demo controls.
* `ClusterBrain_Model/`: Contains the 9 serialized `.pkl` and `.pt` model files organized by layer.
* `Synthetic Datasets/`: Contains the raw synthetic CSV datasets for training and forecasting.
* `requirements.txt`: List of Python dependencies for quick setup.

---

## 🎮 Demo Walkthrough (For Collaborators & Judges)
When testing the dashboard, use the **Sidebar Demo Controls** to trigger the ML pipelines:

1. **The Doctor (TabPFN Predictive Operations):** Select a node from the dropdown and click **"Inject Thermal Anomaly"**. Watch the cluster map turn red, the AI detect the critical 80°C threshold, initiate a theatrical 5-second countdown, and autonomously migrate active jobs to healthy nodes before resetting the burning server!
2. **The Planner (Smart Scheduling & Right Size Proof):** Submit a heavy AI workload (e.g., BERT Fine-tuning) and watch our 4-pass Smart Scheduler pack the nodes. Use the **Interactive Right-Size Proof Simulator** at the bottom of the tab to verify that TabPFN's right-sizing advice is mathematically safe (zero OOM failures) and saves real dollars!
3. **The Accountant:** Click **"Simulate Abandoned Session"** to flag a user session that hasn't executed code in 90 minutes. Use **"Generate User Clusters"** to segment users, and **"Forecast Next Day Cost"** to dynamically pull the latest CSV data and run the XGBoost time-series forecast.

---

## 📜 License & Security Policy

This project is open-source and structured according to standard security protocols.

* 📄 **License:** This project is licensed under the **[Apache 2.0 License](./LICENSE)**.
* 🛡️ **Security Policy:** For reporting vulnerabilities and security guidelines, please see our **[Security Policy](./SECURITY.md)**.
