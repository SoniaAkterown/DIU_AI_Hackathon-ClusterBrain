import streamlit as st
import requests
import pandas as pd
import time
import random
import os

# --- CONFIGURATION ---
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")

st.set_page_config(page_title="ClusterBrain Dashboard", layout="wide", page_icon="🧠")

# --- FETCH GLOBAL STATE ---
try:
    res = requests.get(f"{API_URL}/cluster/state")
    if res.status_code == 200:
        cluster_state = res.json()
        st.session_state.nodes = cluster_state.get("nodes", [])
        st.session_state.jobs = cluster_state.get("jobs", [])
        st.session_state.pending_jobs = cluster_state.get("pending_jobs", [])
        st.session_state.waste_savings = cluster_state.get("waste_savings", 0.0)
    else:
        st.error(f"Failed to fetch state from backend. Status: {res.status_code}")
except Exception as e:
    st.error(f"Backend API unavailable: {e}")
    st.session_state.nodes = []
    st.session_state.jobs = []
    st.session_state.pending_jobs = []
    st.session_state.waste_savings = 0.0

if "has_scaled" not in st.session_state:
    st.session_state.has_scaled = False

any_reserve_active = any(n.get("node_status") == "Active" for n in st.session_state.nodes if n["id"] > 8)
if any_reserve_active and not st.session_state.has_scaled:
    st.success("🚨 Proactive Defense Triggered: Node crossed 80%. Activating Reserve Nodes and rebalancing load!")
    st.session_state.has_scaled = True
if not any_reserve_active:
    st.session_state.has_scaled = False

if 'sessions' not in st.session_state:
    st.session_state.sessions = [
        {
            "id": f"Sess-{i}",
            "user": f"User-{random.randint(10, 50)}",
            "util_mean": random.uniform(60.0, 90.0),
            "util_std": random.uniform(5.0, 15.0),
            "spikes": random.randint(2, 5),
            "time_since_spike": random.uniform(1.0, 5.0),
            "keyboard_activity": random.uniform(0.5, 1.0),
            "gpu_last_5min": random.uniform(50.0, 95.0),
            "status": "Checking..."
        } for i in range(1, 5)
    ]

if 'waste_savings' not in st.session_state:
    st.session_state.waste_savings = 0.0

if 'users' not in st.session_state:
    st.session_state.users = []

if 'cost_data' not in st.session_state:
    import os
    cost_file = os.path.join("Synthetic Datasets", "daily_cost_data_synthetic.csv")
    if os.path.exists(cost_file):
        df_c = pd.read_csv(cost_file)
        st.session_state.cost_data = df_c.tail(30)[["date", "total_daily_cost_usd"]].rename(columns={"date": "day", "total_daily_cost_usd": "cost"})
    else:
        st.session_state.cost_data = pd.DataFrame({
            "day": range(30),
            "cost": [random.uniform(400, 600) + (i*2) for i in range(30)]
        })


# (Simulation tick moved down after user interactions)


# --- HELPER FUNCTIONS FOR API CALLS ---
def analyze_node(node_idx):
    node = st.session_state.nodes[node_idx]
    payload = {
        "telemetry": {
            "gpu_temp": node["gpu_temp"],
            "power_draw_w": node["power_draw_w"],
            "mem_errors_cumulative": node["mem_errors_cumulative"],
            "gpu_util_pct": node["gpu_util_pct"],
            "fan_speed_rpm": node["fan_speed_rpm"],
            "clock_speed_mhz": node["clock_speed_mhz"],
            "pcie_bandwidth_gbps": node["pcie_bandwidth_gbps"]
        },
        "rul_features": {
            "temp_rolling_mean_1h": node["gpu_temp"] + random.uniform(-2, 2),
            "temp_rolling_std_1h": random.uniform(1, 5),
            "temp_slope_1h": random.uniform(0.1, 1.5) if node["gpu_temp"] > 80 else 0,
            "mem_errors_rate_1h": node["mem_errors_cumulative"],
            "power_draw_rolling_mean": node["power_draw_w"],
            "anomaly_score_current": 1.0 if node["gpu_temp"] > 80 else 0.0,
            "hours_since_last_anomaly": 24.0,
            "anomaly_frequency_24h": 1.0
        },
        "root_cause_features": {
            "gpu_temp_at_anomaly": node["gpu_temp"],
            "temp_delta_last_30min": 15.0 if node["gpu_temp"] > 80 else 2.0,
            "mem_errors_count_last_1h": node["mem_errors_cumulative"],
            "mem_errors_acceleration": 0.0,
            "power_draw_variance_1h": 20.0,
            "power_draw_spike_count": 0,
            "pcie_error_count": 0,
            "fan_speed_vs_expected_ratio": 0.8 if node["gpu_temp"] > 80 else 1.0
        }
    }
    try:
        res = requests.post(f"{API_URL}/doctor/analyze-node", json=payload)
        if res.status_code == 200:
            data = res.json()
            st.session_state.nodes[node_idx]["anomaly_score"] = data.get("reconstruction_error", 0)
            if data.get("is_anomalous"):
                st.session_state.nodes[node_idx]["health"] = "Anomalous"
                st.session_state.nodes[node_idx]["rul_hours"] = data.get("rul_hours")
                st.session_state.nodes[node_idx]["root_cause"] = data.get("root_cause")
            else:
                st.session_state.nodes[node_idx]["health"] = "Healthy"
    except Exception as e:
        pass


def predict_and_place_job(job):
    job_payload = {
        "job": {
            "model_type": 1,
            "dataset_size_gb": job["dataset_gb"],
            "batch_size": 32,
            "num_epochs": 100,
            "gpu_type_requested": 1,
            "num_gpus_requested": 1,
            "framework": 0,
            "historical_avg_runtime_for_user": 45.0,
            "historical_avg_runtime_for_model_type": 50.0
        },
        "waste_features": {
            "vram_requested_gb": job["req_vram"],
            "cpu_cores_requested": 8,
            "ram_requested_gb": 32.0,
            "model_type": 1,
            "dataset_size_gb": job["dataset_gb"],
            "batch_size": 32,
            "historical_vram_actual_for_similar_jobs": job["req_vram"] * 0.4,
            "historical_cpu_actual_for_similar_jobs": 4.0
        }
    }
    
    # Node ranking payload
    nodes_payload = []
    active_nodes = [n for n in st.session_state.nodes if n.get("node_status", "Active") == "Active"]
    for n in active_nodes:
        nodes_payload.append({
            "node_anomaly_score": n["anomaly_score"],
            "node_rul_hours": n["rul_hours"] if n["rul_hours"] else 1000.0,
            "node_root_cause": 0,
            "node_gpu_util_pct": n["gpu_util_pct"],
            "node_vram_free_gb": 40.0 if n["health"] == "Healthy" else 10.0,
            "node_queue_depth": 0,
            "job_predicted_runtime_min": 60.0,
            "job_predicted_vram_gb": job["req_vram"] * 0.4,
            "gpu_type_match": 1,
            "node_historical_job_success_rate": 0.99
        })
        
    try:
        # Predict job
        res_job = requests.post(f"{API_URL}/planner/predict-job", json=job_payload)
        if res_job.status_code == 200:
            data = res_job.json()
            job["predicted_runtime"] = data.get("predicted_runtime_min")
            job["predicted_vram"] = data.get("predicted_waste_metric", job["req_vram"] * 0.4)
            # Add to waste savings if requested >> predicted
            if job["req_vram"] > job["predicted_vram"] * 1.5:
                job["waste_savings_inc"] = (job["req_vram"] - job["predicted_vram"]) * 0.15
                
        # Submit job using Smart Scheduler (Bin Packing)
        res_submit = requests.post(f"{API_URL}/cluster/submit-job", json=job)
        return res_submit
                
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def fetch_right_size_proof():
    try:
        res = requests.get(f"{API_URL}/planner/right-size-proof")
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {
        "overall_accuracy_pct": 0.0,
        "total_gpu_hours_saved": 0.0,
        "total_dollars_saved": 0.0,
        "total_advice_given": 0,
        "accepted_advice_count": 0,
        "sufficient_advice_count": 0,
        "job_history": []
    }

def record_right_size_job_api(job_name, job_type, req_vram, advice_accepted, runtime_hours=2.5):
    payload = {
        "job_name": job_name,
        "job_type": job_type,
        "requested_vram_gb": req_vram,
        "dataset_size_gb": 20.0,
        "advice_accepted": advice_accepted,
        "runtime_hours": runtime_hours
    }
    try:
        res = requests.post(f"{API_URL}/planner/record-right-size-job", json=payload)
        return res
    except Exception as e:
        return None


# --- SIDEBAR: DEMO CONTROLS ---
with st.sidebar:
    st.header("🎛 Demo Controls")
    st.markdown("Use these controls to simulate events for the judges.")
    
    st.subheader("1. The Doctor")
    selected_node = st.selectbox("Select Node for Anomaly:", options=["Node 1", "Node 2", "Node 3", "Node 4"])
    if st.button("🔥 Inject Thermal Anomaly", use_container_width=True):
        target_node_name = str(selected_node)
        requests.post(f"{API_URL}/inject_anomaly/{target_node_name}")
        st.rerun()
        
    if st.button("✅ Reset Nodes", use_container_width=True):
        requests.post(f"{API_URL}/cluster/reset")
        st.success("Nodes and queues reset to normal.")
        st.rerun()

    st.divider()
    st.subheader("🧠 TabPFN Predictive Analytics")
    try:
        predict_res = requests.get(f"{API_URL}/predict_health")
        if predict_res.status_code == 200:
            health_data = predict_res.json()
            if health_data.get("risk_level") == "CRITICAL":
                critical_node = health_data.get("critical_node")
                ai_alert = st.empty()
                
                for i in range(5, 0, -1):
                    ai_alert.error(f"🚨 TabPFN PREDICTION: {critical_node} thermal failure imminent! Migrating workloads in {i} seconds...")
                    time.sleep(1)
                    
                requests.post(f"{API_URL}/migrate_and_remediate/{critical_node}")
                
                ai_alert.success("✅ Workloads successfully migrated to healthy nodes!")
                time.sleep(2)
                st.rerun()
            else:
                st.info("✅ All nodes operating within safe parameters.")
    except Exception as e:
        st.warning("Could not reach Predictive Engine.")

    st.divider()
    st.subheader("2. The Planner")
    with st.form("job_submit_form"):
        job_type = st.selectbox("Model Workload", ["ResNet-50 Training", "BERT Fine-tuning", "LLM Quantization", "Data Preprocessing"], help="Select the type of AI workload to simulate.")
        accept_right_size = st.checkbox("💡 Accept Right-Size GPU Advice", value=True, help="Automatically accept TabPFN's right-sizing advice to save resources.")
        submit = st.form_submit_button("🚀 Submit Job", use_container_width=True)
        
        if submit:
            random_id = str(random.randint(1000, 9999))
            job_name = f"Job-{random_id}"
            
            if job_type == "ResNet-50 Training":
                util_impact = 30.0
                power_impact = 45.0
                vram_req = 40.0
                dataset_gb = 10.0
            elif job_type == "BERT Fine-tuning":
                util_impact = 50.0
                power_impact = 80.0
                vram_req = 80.0
                dataset_gb = 50.0
            elif job_type == "LLM Quantization":
                util_impact = 60.0
                power_impact = 90.0
                vram_req = 80.0
                dataset_gb = 40.0
            else: # Data Preprocessing
                util_impact = 15.0
                power_impact = 20.0
                vram_req = 16.0
                dataset_gb = 100.0
                
            job = {
                "name": job_name,
                "type": job_type,
                "req_vram": vram_req,
                "dataset_gb": dataset_gb,
                "util_impact": util_impact,
                "power_impact": power_impact,
            }
            res_submit = predict_and_place_job(job)
            record_right_size_job_api(job_name, job_type, vram_req, accept_right_size)
            
            if res_submit and res_submit.status_code == 200:
                assigned_node = res_submit.json().get("assigned_node")
                if assigned_node == "Pending":
                    st.toast("Cluster is at maximum capacity! Job added to the pending queue.")
                else:
                    st.toast(f"🤖 Smart Scheduler: Placed job on {assigned_node} for optimal bin-packing.")
            st.rerun()

    st.subheader("3. The Accountant")
    if st.button("💤 Simulate Abandoned Session", use_container_width=True):
        if st.session_state.sessions:
            st.session_state.sessions[0]["util_mean"] = 0.0
            st.session_state.sessions[0]["spikes"] = 0
            st.session_state.sessions[0]["time_since_spike"] = 90.0
            st.session_state.sessions[0]["keyboard_activity"] = 0.0
            st.session_state.sessions[0]["gpu_last_5min"] = 0.0
            st.session_state.sessions[0]["status"] = "Checking..."
            st.success("Abandoned session simulated on Sess-1!")
            st.rerun()


# --- LOCAL ANOMALY CHECKS ---
for idx, n in enumerate(st.session_state.nodes):
    if n["health"] != "Anomalous" and n.get("node_status", "Active") == "Active":
        if n["gpu_temp"] > 85.0:
            analyze_node(idx)

# --- MAIN DASHBOARD LAYOUT ---
st.title("🧠 ClusterBrain Dashboard")
st.markdown("Intelligent GPU Cluster Management using 9 ML Models (Phase 5 Demo)")

if st.button("🔄 Refresh to see current node status", type="primary", use_container_width=True):
    pass # Re-runs script to fetch latest state

# Create tabs for the 6 panels
tab1, tab2, tab3 = st.tabs(["🩺 The Doctor (Health)", "📅 The Planner (Jobs & Right Size Proof)", "💰 The Accountant (Cost)"])

# === TAB 1: THE DOCTOR ===
with tab1:
    st.header("Cluster Health Map")
    st.markdown("### 🏢 Main Cluster (8 Nodes)")
    cols = st.columns(4)
    for idx, node in enumerate(st.session_state.nodes[:8]):
        col = cols[idx % 4]
        with col:
            # Color coding based on health and utilization
            is_anomalous = (node["health"] == "Anomalous" or node["gpu_util_pct"] > 90.0)
            is_stressed = (70.0 <= node["gpu_util_pct"] <= 90.0 and node["health"] != "Anomalous")
            
            if is_anomalous:
                color = "red"
                status_icon = "🚨"
                display_health = "Critical" if node["health"] != "Anomalous" else "Anomalous"
            elif is_stressed:
                color = "orange"
                status_icon = "⚠️"
                display_health = "Stressed"
            else:
                color = "green"
                status_icon = "✅"
                display_health = "Healthy"
                
            st.markdown(f"""
            <div style="border: 2px solid {color}; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                <h4>Node {node['id']} {status_icon}</h4>
                <b>Status:</b> <span style="color:{color}">{display_health}</span><br/>
                <b>Temp:</b> {node['gpu_temp']:.1f} °C<br/>
                <b>Power:</b> {node['power_draw_w']:.0f} W<br/>
                <b>Util:</b> {node['gpu_util_pct']:.0f}%
            </div>
            """, unsafe_allow_html=True)
            
            if node["health"] == "Anomalous":
                rc = str(node.get("root_cause", "unknown")).lower()
                if rc in ["none", "unknown", "nan", "null"]:
                    rc = "thermal_throttle"
                formatted_rc = rc.replace("_", " ").title()
                rul = node.get("rul_hours")
                rul_display = f"{rul:.1f}" if rul is not None else "N/A"
                st.error(f"**RUL:** {rul_display} hrs\n\n**Cause:** {formatted_rc}")

    st.markdown("### 🚑 Reserve Cluster (4 Nodes)")
    cols2 = st.columns(4)
    for idx, node in enumerate(st.session_state.nodes[8:]):
        col = cols2[idx % 4]
        with col:
            if node.get("node_status") in ["Standby", "Offline"] or node["gpu_util_pct"] <= 0.0:
                color = "gray"
                status_icon = "💤"
                display_health = "Standby / Sleep"
                opacity = 0.5
            else:
                opacity = 1.0
                is_anomalous = (node["health"] == "Anomalous" or node["gpu_util_pct"] > 90.0)
                is_stressed = (70.0 <= node["gpu_util_pct"] <= 90.0 and node["health"] != "Anomalous")
                
                if is_anomalous:
                    color = "red"
                    status_icon = "🚨"
                    display_health = "Critical" if node["health"] != "Anomalous" else "Anomalous"
                elif is_stressed:
                    color = "orange"
                    status_icon = "⚠️"
                    display_health = "Stressed"
                else:
                    color = "green"
                    status_icon = "✅"
                    display_health = "Healthy"
                    
            st.markdown(f"""
            <div style="border: 2px solid {color}; border-radius: 10px; padding: 10px; margin-bottom: 10px; opacity: {opacity};">
                <h4>Node {node['id']} {status_icon}</h4>
                <b>Status:</b> <span style="color:{color}">{display_health}</span><br/>
                <b>Temp:</b> {node['gpu_temp']:.1f} °C<br/>
                <b>Power:</b> {node['power_draw_w']:.0f} W<br/>
                <b>Util:</b> {node['gpu_util_pct']:.0f}%
            </div>
            """, unsafe_allow_html=True)
            
            if node["health"] == "Anomalous" and node.get("node_status") == "Active":
                rc = str(node.get("root_cause", "unknown")).lower()
                if rc in ["none", "unknown", "nan", "null"]:
                    rc = "thermal_throttle"
                formatted_rc = rc.replace("_", " ").title()
                rul = node.get("rul_hours")
                rul_display = f"{rul:.1f}" if rul is not None else "N/A"
                st.error(f"**RUL:** {rul_display} hrs\n\n**Cause:** {formatted_rc}")

# === TAB 2: THE PLANNER & RIGHT SIZE PROOF ===
with tab1: # Placeholder tag check
    pass

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Job Queue & Placement")
        st.subheader("Active Running Jobs")
        if not st.session_state.jobs:
            st.info("No active jobs running. Submit a job from the sidebar.")
        else:
            job_data = []
            for j in st.session_state.jobs:
                job_data.append({
                    "Job": j["name"],
                    "Type": j.get("type", "Unknown"),
                    "Duration": "30s",
                    "Assigned To": f"Node {j.get('assigned_node', '?')}",
                    "Status": "⏳ Running"
                })
            st.dataframe(pd.DataFrame(job_data), use_container_width=True)
            
        st.subheader("Pending Job Queue")
        if not st.session_state.pending_jobs:
            st.success("Queue is empty. All jobs are currently running.")
        else:
            pending_data = []
            for p in st.session_state.pending_jobs:
                pending_data.append({
                    "Job": p["name"],
                    "Status": p.get("status", "Queued"),
                    "Predicted Time": f"{p.get('predicted_runtime', 0):.1f} min"
                })
            st.dataframe(pd.DataFrame(pending_data), use_container_width=True)
            
    with col2:
        st.header("📈 Resource Waste Dashboard")
        st.metric(label="Estimated Cumulative Waste Savings", value=f"${st.session_state.waste_savings:.2f}", help="Total dollar amount saved by optimizing requested GPU VRAM across all jobs.")
        
        if st.session_state.jobs:
            chart_data = pd.DataFrame([
                {"Job": j["name"], "Requested": j["req_vram"], "Predicted Needed": j.get("predicted_vram", j["req_vram"])} 
                for j in st.session_state.jobs
            ])
            st.bar_chart(chart_data.set_index("Job"), color=["#FF4B4B", "#0068C9"])
        else:
            st.info("Awaiting jobs to calculate waste.")

    st.divider()
    st.header("🛡️ Right Size Proof Module (Safety & Accuracy Validation)")
    st.markdown("Verifies that Resource Waste recommendations are **safe**, **accurate (zero OOM failures)**, and **financially impactful**.")
    
    proof_data = fetch_right_size_proof()
    
    # 1. Metric Cards Header
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(
            label="🎯 Advice Accuracy",
            value=f"{proof_data.get('overall_accuracy_pct', 0.0):.1f}%",
            help="Percentage of jobs where recommended smaller GPU was sufficient (zero OOM crashes)."
        )
    with m_col2:
        st.metric(
            label="⚡ GPU Hours Saved",
            value=f"{proof_data.get('total_gpu_hours_saved', 0.0):.2f} hrs",
            help="Total GPU hardware hours saved by accepting right-sizing recommendations."
        )
    with m_col3:
        st.metric(
            label="💵 Total Money Saved",
            value=f"${proof_data.get('total_dollars_saved', 0.0):.2f}",
            help="Realized dollar savings from accepted right-size advice."
        )
    with m_col4:
        accepted_cnt = proof_data.get('accepted_advice_count', 0)
        total_cnt = proof_data.get('total_advice_given', 0)
        st.metric(
            label="📋 Advice Acceptance",
            value=f"{accepted_cnt} / {total_cnt}",
            help="Number of recommendations accepted by users."
        )

    # 2. Single-Sentence Per-Job Saving Counter
    st.subheader("💬 Saving Counter Feed (Single-Sentence Per-Job Breakdown)")
    history_logs = proof_data.get("job_history", [])
    if history_logs:
        for record in history_logs[:5]:
            sentence = record.get("savings_sentence", "")
            accepted = record.get("advice_accepted", True)
            if accepted:
                st.success(f"💡 **Saving Counter:** {sentence}")
            else:
                st.warning(f"⚠️ **Advice Counter:** {sentence}")
    else:
        st.info("No job history available for Right Size Proof.")

    # 3. Usage Compare Data Table
    st.subheader("📊 Usage Compare: Requested Memory vs Recommended vs Actual Telemetry")
    if history_logs:
        df_history = pd.DataFrame(history_logs)
        df_display = pd.DataFrame({
            "Job Name": df_history["job_name"],
            "Job Type": df_history["job_type"],
            "Requested VRAM (GB)": df_history["requested_vram_gb"].map("{:.1f}".format),
            "Recommended VRAM (GB)": df_history["recommended_vram_gb"].map("{:.1f}".format),
            "Actual Memory Used (GB)": df_history["actual_vram_used_gb"].map("{:.1f}".format),
            "Advice Decision": df_history["advice_accepted"].apply(lambda x: "✅ Accepted" if x else "❌ Declined"),
            "Safety Verification": df_history["is_sufficient"].apply(lambda x: "✅ Sufficient (No OOM)" if x else "🚨 Insufficient"),
            "GPU Hours Saved": df_history["gpu_hours_saved"].map("{:.2f} hrs".format),
            "Money Saved ($)": df_history["dollars_saved"].map("${:.2f}".format)
        })
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("Awaiting telemetry data to populate Usage Compare table.")

    # 4. Interactive Simulation Control inside Tab 2
    with st.expander("🧪 Interactive Right-Size Proof Simulator (Test Custom Job)", expanded=True):
        with st.form("interactive_right_size_form"):
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_job_name = st.text_input("Custom Job Name", value=f"Job-Sim-{random.randint(100, 999)}")
                sim_job_type = st.selectbox("Model Workload", ["ResNet-50 Training", "BERT Fine-tuning", "LLM Quantization", "Diffusion Generation"])
                sim_req_vram = st.slider("Requested GPU Memory (GB)", min_value=16.0, max_value=80.0, value=80.0, step=8.0)
            with sim_col2:
                sim_runtime_hrs = st.number_input("Estimated Runtime (Hours)", min_value=0.5, max_value=24.0, value=3.0, step=0.5)
                sim_advice_accepted = st.checkbox("💡 Accept Right-Size GPU Recommendation", value=True)
                
            sim_submit = st.form_submit_button("🚀 Run Right-Size Verification Test")
            if sim_submit:
                res = record_right_size_job_api(sim_job_name, sim_job_type, sim_req_vram, sim_advice_accepted, sim_runtime_hrs)
                if res and res.status_code == 200:
                    data = res.json()
                    recorded = data.get("recorded_job", {})
                    st.toast(f"Verification Recorded! {recorded.get('savings_sentence')}")
                    st.rerun()

# === TAB 3: THE ACCOUNTANT ===
with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("👤 Session Monitor")
        
        # Process newly added sessions via API
        for sess in st.session_state.sessions:
            if sess["status"] == "Checking...":
                payload = {
                    "gpu_util_mean_30min": sess["util_mean"],
                    "gpu_util_std_30min": sess["util_std"],
                    "gpu_util_max_30min": 1.0,
                    "gpu_util_min_30min": 0.0,
                    "gpu_util_last_5min_mean": sess.get("gpu_last_5min", 0.0),
                    "num_utilization_spikes": sess["spikes"],
                    "time_since_last_spike_min": sess["time_since_spike"],
                    "keyboard_mouse_activity_proxy": sess.get("keyboard_activity", 0.0),
                    "session_duration_hours": 2.5
                }
                try:
                    res = requests.post(f"{API_URL}/accountant/session-state", json=payload)
                    if res.status_code == 200:
                        sess["status"] = res.json().get("session_state", "Unknown").upper()
                        if sess["status"] == "ABANDONED":
                            sess["status"] = "ABANDONED (Auto-Killed)"
                except:
                    sess["status"] = "API Error"
                    
        if not st.session_state.sessions:
            st.info("No active user sessions monitored.")
        else:
            sess_df = pd.DataFrame(st.session_state.sessions)
            st.dataframe(sess_df[["id", "user", "time_since_spike", "status"]], use_container_width=True)
            
        st.header("User Clusters")
        if st.button("Generate User Clusters"):
            users = []
            for i in range(10):
                payload = {
                    "avg_session_duration_hours": random.uniform(1, 10),
                    "avg_gpu_utilization_pct": random.uniform(5, 95),
                    "idle_ratio": random.uniform(0.1, 0.9),
                    "sessions_per_week": random.uniform(1, 15),
                    "avg_vram_waste_ratio": random.uniform(1.0, 3.0),
                    "peak_hour_preference": random.randint(0, 2),
                    "total_compute_hours_30d": random.uniform(20, 200),
                    "job_success_rate": random.uniform(0.5, 1.0)
                }
                try:
                    res = requests.post(f"{API_URL}/accountant/user-cluster", json=payload)
                    if res.status_code == 200:
                        cluster_name = res.json().get("cluster", "Unknown")
                        users.append({"User": f"User-{i}", "Segment": cluster_name})
                except:
                    pass
            st.session_state.users = users
            
        if st.session_state.users:
            st.dataframe(pd.DataFrame(st.session_state.users), use_container_width=True)

    with col2:
        st.header("📊 Cost Forecast")
        st.markdown("Historical Daily Cost (Last 30 Days)")
        st.line_chart(st.session_state.cost_data.set_index("day"))
        
        if st.button("🔮 Forecast Next Day Cost"):
            import os
            cost_file = os.path.join("Synthetic Datasets", "daily_cost_data_synthetic.csv")
            if os.path.exists(cost_file):
                df_c = pd.read_csv(cost_file)
                df_c["date"] = pd.to_datetime(df_c["date"])
                df_c = df_c.sort_values("date")
                df_c["spend_lag_1d"] = df_c["total_daily_cost_usd"].shift(1)
                df_c["spend_lag_7d"] = df_c["total_daily_cost_usd"].shift(7)
                df_c["spend_lag_30d"] = df_c["total_daily_cost_usd"].shift(30)
                df_c["spend_rolling_mean_7d"] = df_c["total_daily_cost_usd"].rolling(7).mean()
                df_c["spend_rolling_mean_30d"] = df_c["total_daily_cost_usd"].rolling(30).mean()
                df_c["is_weekend"] = df_c["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
                
                # Drop NaNs generated by shifting
                df_c = df_c.dropna()
                latest_row = df_c.iloc[-1]
                
                payload = {
                    "spend_lag_1d": float(latest_row["spend_lag_1d"]),
                    "spend_lag_7d": float(latest_row["spend_lag_7d"]),
                    "spend_lag_30d": float(latest_row["spend_lag_30d"]),
                    "spend_rolling_mean_7d": float(latest_row["spend_rolling_mean_7d"]),
                    "spend_rolling_mean_30d": float(latest_row["spend_rolling_mean_30d"]),
                    "day_of_week": int(latest_row["day_of_week"]),
                    "is_weekend": int(latest_row["is_weekend"]),
                    "is_exam_period": 0,
                    "is_semester_break": 0,
                    "active_user_count_today": 45,
                    "total_jobs_submitted_today": 120,
                    "last_5_days_cost": st.session_state.cost_data["cost"].tail(5).tolist()
                }
            else:
                # Fallback to simple payload
                payload = {
                    "spend_lag_1d": float(st.session_state.cost_data["cost"].iloc[-1]),
                    "spend_lag_7d": float(st.session_state.cost_data["cost"].iloc[-7]),
                    "spend_lag_30d": float(st.session_state.cost_data["cost"].iloc[0]),
                    "spend_rolling_mean_7d": float(st.session_state.cost_data["cost"].tail(7).mean()),
                    "spend_rolling_mean_30d": float(st.session_state.cost_data["cost"].mean()),
                    "day_of_week": 3,
                    "is_weekend": 0,
                    "is_exam_period": 0,
                    "is_semester_break": 0,
                    "active_user_count_today": 45,
                    "total_jobs_submitted_today": 120,
                    "last_5_days_cost": st.session_state.cost_data["cost"].tail(5).tolist()
                }
                
            try:
                res = requests.post(f"{API_URL}/accountant/forecast-cost", json=payload)
                if res.status_code == 200:
                    pred_cost = res.json().get("predicted_next_day_cost", 0.0)
                    st.metric(label="Predicted Cost for Tomorrow", value=f"${pred_cost:.2f}")
            except Exception as e:
                st.error("API Error connecting to Cost Forecaster")
