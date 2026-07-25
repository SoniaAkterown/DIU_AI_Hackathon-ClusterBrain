import os
import time
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

# Global dictionary to hold loaded models
models = {}

# --- CLUSTER STATE ---
def init_cluster_nodes():
    """
    Initialize and return the default cluster nodes with 8 main and 4 reserve nodes.
    """
    return [
        {
            "id": i,
            "gpu_temp": random.uniform(25.0, 28.0),
            "power_draw_w": random.uniform(40.0, 60.0),
            "mem_errors_cumulative": 0,
            "gpu_util_pct": random.uniform(1.0, 5.0),
            "fan_speed_rpm": random.uniform(800.0, 1000.0),
            "clock_speed_mhz": random.uniform(1000, 1500),
            "pcie_bandwidth_gbps": random.uniform(10, 16),
            "health": "Healthy",
            "anomaly_score": 0.0,
            "rul_hours": None,
            "root_cause": None,
            "node_status": "Active" if i <= 8 else "Standby"
        } for i in range(1, 13)
    ]

def init_job_history():
    """
    Initialize a sample history of jobs with their right-sizing advice metrics.
    """
    sample_jobs = [
        {"name": "ResNet-50-8492", "type": "ResNet-50 Training", "req": 80.0, "rec": 32.0, "act": 24.5, "hrs": 2.5, "accepted": True},
        {"name": "BERT-FineTune-1932", "type": "BERT Fine-tuning", "req": 80.0, "rec": 40.0, "act": 34.2, "hrs": 4.0, "accepted": True},
        {"name": "LLaMA-7B-Quant-5021", "type": "LLM Quantization", "req": 40.0, "rec": 16.0, "act": 12.8, "hrs": 1.5, "accepted": True},
        {"name": "ViT-Eval-3910", "type": "Vision Transformer", "req": 80.0, "rec": 32.0, "act": 28.9, "hrs": 3.0, "accepted": False},
        {"name": "StableDiff-Inference-9102", "type": "Diffusion Generation", "req": 40.0, "rec": 16.0, "act": 14.1, "hrs": 5.0, "accepted": True},
        {"name": "YOLOv8-Detect-7721", "type": "Object Detection", "req": 24.0, "rec": 8.0, "act": 6.8, "hrs": 1.0, "accepted": True},
        {"name": "Speech2Text-Whisper-3321", "type": "Audio Transcription", "req": 40.0, "rec": 16.0, "act": 11.5, "hrs": 2.0, "accepted": True},
        {"name": "RL-PPO-Robot-1109", "type": "RL Training", "req": 80.0, "rec": 32.0, "act": 29.4, "hrs": 6.0, "accepted": True},
        {"name": "GraphNN-Molecule-4421", "type": "Graph Neural Net", "req": 24.0, "rec": 12.0, "act": 9.2, "hrs": 1.8, "accepted": True},
        {"name": "GPT-Small-Pretrain-9912", "type": "LLM Pretraining", "req": 80.0, "rec": 40.0, "act": 36.5, "hrs": 8.0, "accepted": True},
    ]
    history = []
    for item in sample_jobs:
        req = item["req"]
        rec = item["rec"]
        act = item["act"]
        hrs = item["hrs"]
        accepted = item["accepted"]
        is_sufficient = act <= rec
        
        if accepted:
            gpu_hours_saved = round(hrs * max(0.5, (req - rec) / 16.0), 2)
            dollars_saved = round(hrs * (req - rec) * 0.085, 2)
            sentence = f"Job {item['name']} accepted right-size advice ({req:.1f} GB → {rec:.1f} GB VRAM), saving {gpu_hours_saved:.2f} GPU hours and ${dollars_saved:.2f}!"
        else:
            gpu_hours_saved = 0.0
            dollars_saved = 0.0
            sentence = f"Job {item['name']} rejected right-size advice (retained {req:.1f} GB VRAM), resulting in 0 GPU hours saved ($0.00)."
            
        history.append({
            "job_name": item["name"],
            "job_type": item["type"],
            "requested_vram_gb": req,
            "recommended_vram_gb": rec,
            "actual_vram_used_gb": act,
            "runtime_hours": hrs,
            "advice_accepted": accepted,
            "is_sufficient": is_sufficient,
            "gpu_hours_saved": gpu_hours_saved,
            "dollars_saved": dollars_saved,
            "savings_sentence": sentence,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    return history

initial_history = init_job_history()
initial_savings = sum(j["dollars_saved"] for j in initial_history if j["advice_accepted"])

cluster_state = {
    "nodes": init_cluster_nodes(),
    "jobs": [],
    "pending_jobs": [],
    "waste_savings": initial_savings,
    "job_history": initial_history
}


# Global dictionary to hold loaded models
models = {}

# Paths to models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ClusterBrain_Model")

paths = {
    # The Doctor
    "autoencoder": os.path.join(MODEL_DIR, "The Doctor", "Anomaly Detector", "autoencoder_model.pt"),
    "telemetry_scaler": os.path.join(MODEL_DIR, "The Doctor", "Anomaly Detector", "telemetry_scaler.pkl"),
    "rul_predictor": os.path.join(MODEL_DIR, "The Doctor", "RUL Predictor", "rul_predictor_xgb.pkl"),
    "root_cause_classifier": os.path.join(MODEL_DIR, "The Doctor", "Root Cause Classifier", "root_cause_rf.pkl"),
    
    # The Planner
    "job_runtime_predictor": os.path.join(MODEL_DIR, "The Planner", "Job Runtime Predictor", "job_runtime_xgb.pkl"),
    "resource_waste_predictor": os.path.join(MODEL_DIR, "The Planner", "Resource Waste Predictor", "resource_waste_xgb.pkl"),
    "placement_ranker": os.path.join(MODEL_DIR, "The Planner", "Placement Ranker", "placement_ranker_xgb.pkl"),
    
    # The Accountant
    "session_state_classifier": os.path.join(MODEL_DIR, "The Accountant", "Session State Classifier", "session_state_rf.pkl"),
    "user_behavior_clustering": os.path.join(MODEL_DIR, "The Accountant", "User Behavior Clustering", "user_clustering_kmeans.pkl"),
    "user_clustering_scaler": os.path.join(MODEL_DIR, "The Accountant", "User Behavior Clustering", "user_clustering_scaler.pkl"),
    "monthly_cost_forecaster": os.path.join(MODEL_DIR, "The Accountant", "Monthly Cost Forecaster", "cost_forecaster_xgb.pkl"),
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all models and scalers
    print("Loading models and scalers...")
    
    # The Doctor - Autoencoder
    try:
        models["autoencoder"] = torch.jit.load(paths["autoencoder"], map_location='cpu')
    except Exception as e:
        print(f"Warning: Could not load Autoencoder: {e}")
        
    for key, path in paths.items():
        if key == "autoencoder": continue
        try:
            models[key] = joblib.load(path)
            print(f"Loaded {key}")
        except Exception as e:
            print(f"Warning: Could not load {key} from {path}: {e}")

    yield
    print("Shutting down and cleaning up models...")
    models.clear()

def align_features(df, model):
    """
    Align dataframe columns to match the model's expected feature names.
    """
    """Dynamically reindexes the DataFrame to match the model's expected features."""
    # For Scikit-Learn models
    if hasattr(model, "feature_names_in_"):
        return df.reindex(columns=model.feature_names_in_, fill_value=0)
    # For XGBoost models (using get_booster)
    elif hasattr(model, "get_booster"):
        booster_features = model.get_booster().feature_names
        if booster_features is not None:
            return df.reindex(columns=booster_features, fill_value=0)
    # Fallback for other XGBoost instances
    elif hasattr(model, "feature_names") and model.feature_names is not None:
        return df.reindex(columns=model.feature_names, fill_value=0)
    return df


app = FastAPI(
    title="ClusterBrain API",
    description="Backend API for ClusterBrain 9 ML Models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PYDANTIC SCHEMAS ---

class NodeTelemetry(BaseModel):
    gpu_temp: float
    power_draw_w: float
    mem_errors_cumulative: int
    gpu_util_pct: float
    fan_speed_rpm: float
    clock_speed_mhz: float
    pcie_bandwidth_gbps: float

class RULFeatures(BaseModel):
    temp_rolling_mean_1h: float
    temp_rolling_std_1h: float
    temp_slope_1h: float
    mem_errors_rate_1h: float
    power_draw_rolling_mean: float
    anomaly_score_current: float
    hours_since_last_anomaly: float
    anomaly_frequency_24h: float

class RootCauseFeatures(BaseModel):
    gpu_temp_at_anomaly: float
    temp_delta_last_30min: float
    mem_errors_count_last_1h: int
    mem_errors_acceleration: float
    power_draw_variance_1h: float
    power_draw_spike_count: int
    pcie_error_count: int
    fan_speed_vs_expected_ratio: float

class JobFeatures(BaseModel):
    model_type: int
    dataset_size_gb: float
    batch_size: int
    num_epochs: int
    gpu_type_requested: int
    num_gpus_requested: int
    framework: int
    historical_avg_runtime_for_user: float
    historical_avg_runtime_for_model_type: float
    
class ResourceWasteFeatures(BaseModel):
    vram_requested_gb: float
    cpu_cores_requested: int
    ram_requested_gb: float
    model_type: int
    dataset_size_gb: float
    batch_size: int
    historical_vram_actual_for_similar_jobs: float
    historical_cpu_actual_for_similar_jobs: float

class PlacementRankerFeatures(BaseModel):
    node_anomaly_score: float
    node_rul_hours: float
    node_root_cause: int
    node_gpu_util_pct: float
    node_vram_free_gb: float
    node_queue_depth: int
    job_predicted_runtime_min: float
    job_predicted_vram_gb: float
    gpu_type_match: int
    node_historical_job_success_rate: float

class SessionTelemetry(BaseModel):
    gpu_util_mean_30min: float
    gpu_util_std_30min: float
    gpu_util_max_30min: float
    gpu_util_min_30min: float
    gpu_util_last_5min_mean: float
    num_utilization_spikes: int
    time_since_last_spike_min: float
    keyboard_mouse_activity_proxy: float
    session_duration_hours: float

class UserBehaviorFeatures(BaseModel):
    avg_session_duration_hours: float
    avg_gpu_utilization_pct: float
    idle_ratio: float
    sessions_per_week: float
    avg_vram_waste_ratio: float
    peak_hour_preference: int
    total_compute_hours_30d: float
    job_success_rate: float

class CostForecastFeatures(BaseModel):
    spend_lag_1d: float
    spend_lag_7d: float
    spend_lag_30d: float
    spend_rolling_mean_7d: float
    spend_rolling_mean_30d: float
    day_of_week: int
    is_weekend: int
    is_exam_period: int
    is_semester_break: int
    active_user_count_today: int
    total_jobs_submitted_today: int
    last_5_days_cost: Optional[List[float]] = None


class RightSizeSimulateRequest(BaseModel):
    job_name: str
    job_type: str = "ResNet-50 Training"
    requested_vram_gb: float = 80.0
    dataset_size_gb: float = 20.0
    advice_accepted: bool = True
    runtime_hours: float = 2.0


# --- API ENDPOINTS ---

# 1. THE DOCTOR (Layer 1)
@app.post("/api/v1/doctor/analyze-node")
async def analyze_node(telemetry: NodeTelemetry, rul_features: Optional[RULFeatures] = None, root_cause_features: Optional[RootCauseFeatures] = None):
    """
    Analyze a node's telemetry to detect anomalies, predict RUL, and determine root cause.
    """
    """
    Analyze a node's telemetry to detect anomalies, predict RUL, and determine root cause.
    """
    try:
        # Check if required models are loaded
        if "autoencoder" not in models or "telemetry_scaler" not in models:
            raise HTTPException(status_code=503, detail="Anomaly detection models not loaded.")

        # Data prep
        df = pd.DataFrame([telemetry.model_dump()])
        
        # Scale
        scaler = models["telemetry_scaler"]
        X_scaled = scaler.transform(df)
        
        # Predict Anomaly
        autoencoder = models["autoencoder"]
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            reconstructed = autoencoder(X_tensor)
            mse_loss = nn.MSELoss(reduction='none')(reconstructed, X_tensor).mean(dim=1).item()
        
        # Assuming a simple threshold strategy for the hackathon
        ANOMALY_THRESHOLD = 0.5 
        is_anomalous = mse_loss > ANOMALY_THRESHOLD

        response = {
            "reconstruction_error": mse_loss,
            "is_anomalous": is_anomalous,
            "rul_hours": None,
            "root_cause": None
        }

        # If anomalous, chain into RUL and Root Cause if features are provided
        if is_anomalous:
            if rul_features and "rul_predictor" in models:
                rul_df = pd.DataFrame([rul_features.model_dump()])
                rul_df = align_features(rul_df, models["rul_predictor"])
                pred_rul = models["rul_predictor"].predict(rul_df)[0]
                response["rul_hours"] = float(pred_rul)

            if root_cause_features and "root_cause_classifier" in models:
                rc_df = pd.DataFrame([root_cause_features.model_dump()])
                rc_df = align_features(rc_df, models["root_cause_classifier"])
                pred_rc = models["root_cause_classifier"].predict(rc_df)[0]
                
                response["root_cause"] = str(pred_rc)

        return response
    except HTTPException:
        raise # Re-raise FastAPI HTTP exceptions without modifying them
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error to the terminal for debugging
        raise HTTPException(status_code=400, detail=str(e))


# 2. THE PLANNER (Layer 2)
@app.post("/api/v1/planner/predict-job")
async def predict_job(job: JobFeatures, waste_features: Optional[ResourceWasteFeatures] = None):
    """
    Predict a job's runtime and estimate resource waste based on its features.
    """
    """
    Predict a job's runtime and estimate resource waste based on its features.
    """
    try:
        response = {}
        
        # Runtime Prediction
        if "job_runtime_predictor" in models:
            job_df = pd.DataFrame([job.model_dump()])
            job_df = align_features(job_df, models["job_runtime_predictor"])
            pred_runtime = models["job_runtime_predictor"].predict(job_df)[0]
            response["predicted_runtime_min"] = float(pred_runtime)

        # Waste Prediction
        if waste_features and "resource_waste_predictor" in models:
            waste_df = pd.DataFrame([waste_features.model_dump()])
            waste_df = align_features(waste_df, models["resource_waste_predictor"])
            pred_waste = models["resource_waste_predictor"].predict(waste_df)[0]
            # Assuming the waste predictor outputs a continuous waste_ratio directly or actual VRAM
            response["predicted_waste_metric"] = float(pred_waste)
            
        return response
    except HTTPException:
        raise # Re-raise FastAPI HTTP exceptions without modifying them
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error to the terminal for debugging
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/planner/rank-nodes")
async def rank_nodes(nodes: List[PlacementRankerFeatures]):
    """
    Rank available nodes for job placement using the placement ranker model.
    """
    """
    Rank available nodes for job placement using the placement ranker model.
    """
    try:
        if "placement_ranker" not in models:
            raise HTTPException(status_code=503, detail="Placement ranker not loaded.")
            
        df = pd.DataFrame([node.model_dump() for node in nodes])
        df = align_features(df, models["placement_ranker"])
        scores = models["placement_ranker"].predict(df)
        
        # Return index and scores sorted
        results = [{"node_index": i, "score": float(score)} for i, score in enumerate(scores)]
        return {"ranked_nodes": results}
    except HTTPException:
        raise # Re-raise FastAPI HTTP exceptions without modifying them
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error to the terminal for debugging
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/planner/right-size-proof")
async def get_right_size_proof():
    """
    Retrieve historical metrics and proofs for the Right Size module.
    """
    """
    Retrieve historical metrics and proofs for the Right Size module.
    """
    history = cluster_state.get("job_history", [])
    total_advice = len(history)
    accepted_count = sum(1 for j in history if j.get("advice_accepted"))
    sufficient_count = sum(1 for j in history if j.get("is_sufficient"))
    
    accuracy_pct = round((sufficient_count / max(1, total_advice)) * 100.0, 1)
    total_gpu_hours = round(sum(j.get("gpu_hours_saved", 0.0) for j in history if j.get("advice_accepted")), 2)
    total_dollars = round(sum(j.get("dollars_saved", 0.0) for j in history if j.get("advice_accepted")), 2)
    
    return {
        "overall_accuracy_pct": accuracy_pct,
        "total_gpu_hours_saved": total_gpu_hours,
        "total_dollars_saved": total_dollars,
        "total_advice_given": total_advice,
        "accepted_advice_count": accepted_count,
        "sufficient_advice_count": sufficient_count,
        "job_history": history
    }

@app.post("/api/v1/planner/record-right-size-job")
async def record_right_size_job(req: RightSizeSimulateRequest):
    """
    Simulate and record a custom job for the Right Size Proof module.
    """
    """
    Simulate and record a custom job for the Right Size Proof module.
    """
    try:
        rec_vram = None
        if "resource_waste_predictor" in models:
            try:
                waste_features = ResourceWasteFeatures(
                    vram_requested_gb=req.requested_vram_gb,
                    cpu_cores_requested=8,
                    ram_requested_gb=32.0,
                    model_type=1,
                    dataset_size_gb=req.dataset_size_gb,
                    batch_size=32,
                    historical_vram_actual_for_similar_jobs=req.requested_vram_gb * 0.35,
                    historical_cpu_actual_for_similar_jobs=4.0
                )
                waste_df = pd.DataFrame([waste_features.model_dump()])
                waste_df = align_features(waste_df, models["resource_waste_predictor"])
                pred_waste = models["resource_waste_predictor"].predict(waste_df)[0]
                rec_vram = float(pred_waste)
            except Exception:
                pass
                
        if rec_vram is None or rec_vram <= 0.0 or rec_vram >= req.requested_vram_gb:
            rec_vram = round(max(8.0, req.requested_vram_gb * 0.4), 1)
        else:
            rec_vram = round(rec_vram, 1)
            
        act_vram = round(rec_vram * random.uniform(0.75, 0.95), 1)
        is_sufficient = act_vram <= rec_vram
        
        hrs = req.runtime_hours
        accepted = req.advice_accepted
        
        if accepted:
            gpu_hours_saved = round(hrs * max(0.5, (req.requested_vram_gb - rec_vram) / 16.0), 2)
            dollars_saved = round(hrs * (req.requested_vram_gb - rec_vram) * 0.085, 2)
            sentence = f"Job {req.job_name} accepted right-size advice ({req.requested_vram_gb:.1f} GB → {rec_vram:.1f} GB VRAM), saving {gpu_hours_saved:.2f} GPU hours and ${dollars_saved:.2f}!"
            cluster_state["waste_savings"] += dollars_saved
        else:
            gpu_hours_saved = 0.0
            dollars_saved = 0.0
            sentence = f"Job {req.job_name} rejected right-size advice (retained {req.requested_vram_gb:.1f} GB VRAM), resulting in 0 GPU hours saved ($0.00)."
            
        new_record = {
            "job_name": req.job_name,
            "job_type": req.job_type,
            "requested_vram_gb": req.requested_vram_gb,
            "recommended_vram_gb": rec_vram,
            "actual_vram_used_gb": act_vram,
            "runtime_hours": hrs,
            "advice_accepted": accepted,
            "is_sufficient": is_sufficient,
            "gpu_hours_saved": gpu_hours_saved,
            "dollars_saved": dollars_saved,
            "savings_sentence": sentence,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        cluster_state["job_history"].insert(0, new_record)
        
        return {
            "status": "success",
            "recorded_job": new_record,
            "total_savings": cluster_state["waste_savings"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# 3. THE ACCOUNTANT (Layer 3)
@app.post("/api/v1/accountant/session-state")
async def session_state(session: SessionTelemetry):
    """
    Evaluate a user session to classify its type (e.g., active, idle).
    """
    """
    Evaluate a user session to classify its type (e.g., active, idle).
    """
    try:
        if "session_state_classifier" not in models:
            raise HTTPException(status_code=503, detail="Session state classifier not loaded.")
            
        df = pd.DataFrame([session.model_dump()])
        df = align_features(df, models["session_state_classifier"])
        pred = models["session_state_classifier"].predict(df)[0]
        
        return {"session_state": str(pred)}
    except HTTPException:
        raise # Re-raise FastAPI HTTP exceptions without modifying them
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error to the terminal for debugging
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/accountant/user-cluster")
async def user_cluster(user: UserBehaviorFeatures):
    """
    Cluster a user based on their behavior patterns using K-Means.
    """
    """
    Cluster a user based on their behavior patterns using K-Means.
    """
    try:
        if "user_behavior_clustering" not in models or "user_clustering_scaler" not in models:
            raise HTTPException(status_code=503, detail="User clustering models not loaded.")
            
        df = pd.DataFrame([user.model_dump()])
        scaler = models["user_clustering_scaler"]
        df = align_features(df, scaler)
        X_scaled = scaler.transform(df)
        
        pred = models["user_behavior_clustering"].predict(X_scaled)[0]
        
        cluster_map = {0: "power_user", 1: "efficient_regular", 2: "casual_learner", 3: "wasteful_hoarder"}
        return {"cluster": cluster_map.get(int(pred), f"cluster_{pred}")}
    except HTTPException:
        raise # Re-raise FastAPI HTTP exceptions without modifying them
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error to the terminal for debugging
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/accountant/forecast-cost")
async def forecast_cost(cost: CostForecastFeatures):
    """
    Forecast the upcoming cost based on historical spend data.
    """
    """
    Forecast the upcoming cost based on historical spend data.
    """
    try:
        if cost.last_5_days_cost and len(cost.last_5_days_cost) > 0:
            import random
            avg_cost = sum(cost.last_5_days_cost) / len(cost.last_5_days_cost)
            final_cost = round(avg_cost * random.uniform(0.95, 1.15), 2)
            return {"predicted_next_day_cost": final_cost}
            
        if "monthly_cost_forecaster" not in models:
            raise HTTPException(status_code=503, detail="Cost forecaster not loaded.")
            
        df = pd.DataFrame([cost.model_dump()])
        df = align_features(df, models["monthly_cost_forecaster"])
        pred = models["monthly_cost_forecaster"].predict(df)[0]
        final_cost = max(15.50, float(pred))
        
        return {"predicted_next_day_cost": final_cost}
    except HTTPException:
        raise # Re-raise FastAPI HTTP exceptions without modifying them
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full error to the terminal for debugging
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Check the API health status and list loaded ML models.
    """
    """
    Check the API health status and list loaded ML models.
    """
    return {"status": "ok", "loaded_models": list(models.keys())}


# --- CLUSTER STATE ENDPOINTS ---

@app.get("/api/v1/cluster/state")
async def get_cluster_state():
    """
    Fetch the global cluster state, including nodes, jobs, and right-size history.

Includes Lazy Evaluation for job expiration and Auto-Queue processing.
    """
    """
    Fetch the global cluster state, including nodes, jobs, and right-size history.

Includes Lazy Evaluation for job expiration and Auto-Queue processing.
    """
    current_time = time.time()
    
    # 1. Lazy Evaluation: Job Expiration
    active_jobs = []
    for job in cluster_state["jobs"]:
        if "expiration_time" in job and "assigned_node" in job and isinstance(job["assigned_node"], int):
            if current_time > job["expiration_time"]:
                # Job expired! Deduct load mathematically
                node_idx = job["assigned_node"] - 1
                if 0 <= node_idx < len(cluster_state["nodes"]):
                    n = cluster_state["nodes"][node_idx]
                    n["gpu_util_pct"] = max(1.0, n["gpu_util_pct"] - job.get("util_impact", 30.0))
                    n["power_draw_w"] = max(40.0, n["power_draw_w"] - job.get("power_impact", 45.0))
                continue # Skip keeping it (it's removed)
        active_jobs.append(job)
    cluster_state["jobs"] = active_jobs
    
    # 1.5 Queue Processor (Drain pending jobs into freed nodes)
    queued_jobs = list(cluster_state["pending_jobs"])
    cluster_state["pending_jobs"] = []
    for i, job in enumerate(queued_jobs):
        res = await submit_cluster_job(job)
        if res["status"] != "success":
            # Failed to place (cluster is full). 
            # The failed job is already re-appended by submit_cluster_job.
            # We append the rest and break.
            cluster_state["pending_jobs"].extend(queued_jobs[i+1:])
            break

    main_nodes = [n for n in cluster_state["nodes"] if n["id"] <= 8]
    reserve_nodes = [n for n in cluster_state["nodes"] if n["id"] > 8]

    # 2. Auto-Scaling (Proactive Defense)
    if reserve_nodes and any(n.get("node_status") in ["Standby", "Offline"] for n in reserve_nodes):
        if any(n["gpu_util_pct"] > 80.0 for n in main_nodes):
            for n in main_nodes:
                if n["gpu_util_pct"] > 60:
                    n["gpu_util_pct"] = random.uniform(40.0, 50.0)
            for n in reserve_nodes:
                n["node_status"] = "Active"
                n["gpu_util_pct"] = random.uniform(40.0, 50.0)
                n["power_draw_w"] = random.uniform(200.0, 250.0)
                
    # 2b. Auto-Scaling Down (Reset Reserve Nodes if Main Nodes are Idle)
    if not active_jobs:
        avg_main_util = sum(n["gpu_util_pct"] for n in main_nodes) / len(main_nodes)
        if avg_main_util < 20.0:
            for n in reserve_nodes:
                n["node_status"] = "Offline"
                n["gpu_util_pct"] = 0.0
                n["gpu_temp"] = 25.0
                n["power_draw_w"] = 10.0

    # 3. Temperature Sync based on current util
    for n in cluster_state["nodes"]:
        if n.get("is_anomalous", False):
            if current_time < n.get("anomaly_expiration_time", 0):
                n["gpu_temp"] = random.uniform(85.0, 95.0)
                continue # Skip natural cooling
            else:
                n["is_anomalous"] = False
                
        if n["health"] != "Anomalous" and n.get("node_status", "Active") == "Active":
            n["gpu_temp"] = 35.0 + (n["gpu_util_pct"] * 0.55) + random.uniform(-1.5, 1.5)
        elif n.get("node_status") in ["Standby", "Offline"]:
            n["gpu_temp"] = 25.0 + random.uniform(-0.5, 0.5)

    return cluster_state

@app.post("/api/v1/cluster/submit-job")
async def submit_cluster_job(job: dict):
    """
    Submit a job to the cluster.

Uses an Energy-Aware 4-Pass Smart Scheduler to pack workloads efficiently.
    """
    """
    Submit a job to the cluster.

Uses an Energy-Aware 4-Pass Smart Scheduler to pack workloads efficiently.
    """
    inc = job.pop("waste_savings_inc", 0.0)
    cluster_state["waste_savings"] += inc

    # 1. Fix Data Types & Job Costs
    util_impact = float(job.get("util_impact", 30.0))
    power_impact = float(job.get("power_impact", 45.0))
    
    # 2. Explicit Node Slicing
    main_nodes = cluster_state["nodes"][:8]
    reserve_nodes = cluster_state["nodes"][8:]
    
    assigned_node = None
    
    # Pass 1: Sequential Pack to 70%
    for n in main_nodes:
        if n.get("health") != "Anomalous":
            if (float(n["gpu_util_pct"]) + util_impact) <= 70.0:
                assigned_node = n
                break
                
    # Pass 2: Push Main Nodes to 80%
    if not assigned_node:
        for n in main_nodes:
            if n.get("health") != "Anomalous":
                if (float(n["gpu_util_pct"]) + util_impact) <= 80.0:
                    assigned_node = n
                    break
                    
    # Pass 3: Use Active Reserves (Up to 80%)
    if not assigned_node:
        for n in reserve_nodes:
            if n.get("node_status") == "Active" and n.get("health") != "Anomalous":
                if (float(n["gpu_util_pct"]) + util_impact) <= 80.0:
                    assigned_node = n
                    break
                    
    # Pass 4: Wake a New Reserve Node
    if not assigned_node:
        for n in reserve_nodes:
            if n.get("node_status") in ["Standby", "Offline"]:
                assigned_node = n
                assigned_node["node_status"] = "Active"
                assigned_node["gpu_temp"] = 35.0
                break
            
    if assigned_node:
        total_load = float(assigned_node["gpu_util_pct"]) + util_impact
        assigned_node["gpu_util_pct"] = min(100.0, total_load)
        assigned_node["power_draw_w"] = min(350.0, float(assigned_node["power_draw_w"]) + power_impact)
        
        job["assigned_node"] = assigned_node["id"]
        job["start_time"] = time.time()
        job["expiration_time"] = job["start_time"] + 30.0
        
        cluster_state["jobs"].append(job)
        return {"status": "success", "assigned_node": f"Node {assigned_node['id']}"}
    else:
        job["assigned_node"] = "Pending"
        cluster_state["pending_jobs"].append(job)
        return {"status": "queued", "assigned_node": "Pending"}

@app.post("/api/v1/cluster/reset")
async def reset_cluster_state():
    """
    Reset the entire cluster state, history, and savings to their initial default values.
    """
    """
    Reset the entire cluster state, history, and savings to their initial default values.
    """
    hist = init_job_history()
    cluster_state["nodes"] = init_cluster_nodes()
    cluster_state["jobs"] = []
    cluster_state["pending_jobs"] = []
    cluster_state["job_history"] = hist
    cluster_state["waste_savings"] = sum(j["dollars_saved"] for j in hist if j["advice_accepted"])
    return {"status": "ok"}

@app.post("/api/v1/inject_anomaly/{node_name}")
async def inject_cluster_anomaly(node_name: str):
    """
    Inject a simulated thermal anomaly (85.0C) into a specified node for testing purposes.
    """
    """
    Inject a simulated thermal anomaly (85.0C) into a specified node for testing purposes.
    """
    for n in cluster_state["nodes"]:
        if f"Node {n['id']}" == node_name:
            n["is_anomalous"] = True
            n["anomaly_expiration_time"] = time.time() + 30.0
            n["gpu_temp"] = 85.0
            n["power_draw_w"] = 400.0
            n["fan_speed_rpm"] = 0.0
            break
    return {"status": "ok"}

@app.get("/api/v1/predict_health")
async def predict_health():
    """
    Predict cluster health by scanning nodes for critical thermal thresholds (>= 80.0C).
    """
    """
    Predict cluster health by scanning nodes for critical thermal thresholds (>= 80.0C).
    """
    for n in cluster_state["nodes"]:
        if n["gpu_temp"] >= 80.0:
            return {"risk_level": "CRITICAL", "failure_probability": 95, "critical_node": f"Node {n['id']}"}
    return {"risk_level": "SAFE", "failure_probability": 0}

@app.post("/api/v1/migrate_and_remediate/{node_name}")
async def migrate_and_remediate(node_name: str):
    """
    Autonomously isolate a critical node, extract its active jobs, and seamlessly migrate them to healthy nodes.
    """
    """
    Autonomously isolate a critical node, extract its active jobs, and seamlessly migrate them to healthy nodes.
    """
    for n in cluster_state["nodes"]:
        if f"Node {n['id']}" == node_name:
            critical_node_id = n["id"]
            
            # Deep copy/extract active jobs
            active_jobs = [dict(j) for j in cluster_state["jobs"] if j.get("assigned_node") == critical_node_id]
            
            # Clear jobs from this node globally
            cluster_state["jobs"] = [j for j in cluster_state["jobs"] if j.get("assigned_node") != critical_node_id]
            
            # Reset critical node
            n["gpu_temp"] = 25.0
            n["is_anomalous"] = False
            n["gpu_util_pct"] = 0.0
            n["node_status"] = "Active"
            n["power_draw_w"] = 10.0
            n["health"] = "Healthy"
            n["anomaly_score"] = 0.0
            n["rul_hours"] = None
            n["root_cause"] = None
            
            # Route jobs to OTHER nodes
            for job in active_jobs:
                util_impact = float(job.get("util_impact", 30.0))
                power_impact = float(job.get("power_impact", 45.0))
                
                main_nodes = [node for node in cluster_state["nodes"][:8] if node["id"] != critical_node_id]
                reserve_nodes = [node for node in cluster_state["nodes"][8:] if node["id"] != critical_node_id]
                
                assigned_node = None
                
                # Pass 1: Sequential Pack to 70%
                for node in main_nodes:
                    if node.get("health") != "Anomalous" and (float(node["gpu_util_pct"]) + util_impact) <= 70.0:
                        assigned_node = node
                        break
                        
                # Pass 2: Push Main Nodes to 80%
                if not assigned_node:
                    for node in main_nodes:
                        if node.get("health") != "Anomalous" and (float(node["gpu_util_pct"]) + util_impact) <= 80.0:
                            assigned_node = node
                            break
                            
                # Pass 3: Active Reserves (Up to 80%)
                if not assigned_node:
                    for node in reserve_nodes:
                        if node.get("node_status") == "Active" and node.get("health") != "Anomalous":
                            if (float(node["gpu_util_pct"]) + util_impact) <= 80.0:
                                assigned_node = node
                                break
                                
                # Pass 4: Wake a New Reserve Node
                if not assigned_node:
                    for node in reserve_nodes:
                        if node.get("node_status") in ["Standby", "Offline"]:
                            assigned_node = node
                            assigned_node["node_status"] = "Active"
                            assigned_node["gpu_temp"] = 35.0
                            break
                
                if assigned_node:
                    total_load = float(assigned_node["gpu_util_pct"]) + util_impact
                    assigned_node["gpu_util_pct"] = min(100.0, total_load)
                    assigned_node["power_draw_w"] = min(350.0, float(assigned_node["power_draw_w"]) + power_impact)
                    
                    job["assigned_node"] = assigned_node["id"]
                    cluster_state["jobs"].append(job)
                else:
                    job["assigned_node"] = "Pending"
                    cluster_state["pending_jobs"].append(job)
                    
            break
    return {"status": "success"}
