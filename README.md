# 🧠 ClusterBrain

**ClusterBrain** is an intelligent, machine-learning-driven GPU Cluster Management System. Built for the **DIU AI Hackathon**, this project aims to predict hardware failures, intelligently schedule workloads to minimize resource waste, and forecast cluster operational costs using a robust 3-layer architecture comprising 9 specialized ML models.

## [Presentation Slide](https://docs.google.com/presentation/d/1A5wgFCN63JiA5eeAfuJiWww_NYu1PuhZ/edit?usp=sharing&ouid=108385811874463880409&rtpof=true&sd=true](https://drive.google.com/drive/folders/1rNY_AcA-xKjafIzE0mKQ8O6cBlzFp-uJ?usp=sharing)

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

## ⚠️ Troubleshooting & Known Issues
* **macOS XGBoost Threading Crash:** If the API crashes when predicting on macOS, do not worry. `app.py` already includes `os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'` and `OMP_NUM_THREADS = '1'` at the very top of the file to bypass Apple's threading conflicts with OpenMP. Do not remove these lines.
* **PyTorch Security Warnings:** The autoencoder is saved as a TorchScript archive and is loaded securely using `torch.jit.load()`.
* **Feature Names Mismatch:** All endpoints utilize a dynamic `align_features(df, model)` helper function in `app.py` to guarantee that frontend payloads precisely match the columns expected by Scikit-Learn and XGBoost.
