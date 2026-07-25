import re

with open('app.py', 'r') as f:
    content = f.read()

def inject_docstring(func_name, docstring):
    global content
    pattern = r"(async def " + func_name + r"\(.*?\):)\n"
    replacement = r'\1\n    """\n    ' + docstring + r'\n    """\n'
    content = re.sub(pattern, replacement, content)
    
    pattern_sync = r"(def " + func_name + r"\(.*?\):)\n"
    replacement_sync = r'\1\n    """\n    ' + docstring + r'\n    """\n'
    content = re.sub(pattern_sync, replacement_sync, content)

inject_docstring("init_cluster_nodes", "Initialize and return the default cluster nodes with 8 main and 4 reserve nodes.")
inject_docstring("init_job_history", "Initialize a sample history of jobs with their right-sizing advice metrics.")
inject_docstring("align_features", "Align dataframe columns to match the model's expected feature names.")
inject_docstring("analyze_node", "Analyze a node's telemetry to detect anomalies, predict RUL, and determine root cause.")
inject_docstring("predict_job", "Predict a job's runtime and estimate resource waste based on its features.")
inject_docstring("rank_nodes", "Rank available nodes for job placement using the placement ranker model.")
inject_docstring("get_right_size_proof", "Retrieve historical metrics and proofs for the Right Size module.")
inject_docstring("record_right_size_job", "Simulate and record a custom job for the Right Size Proof module.")
inject_docstring("session_state", "Evaluate a user session to classify its type (e.g., active, idle).")
inject_docstring("user_cluster", "Cluster a user based on their behavior patterns using K-Means.")
inject_docstring("forecast_cost", "Forecast the upcoming cost based on historical spend data.")
inject_docstring("health_check", "Check the API health status and list loaded ML models.")
inject_docstring("get_cluster_state", "Fetch the global cluster state, including nodes, jobs, and right-size history.\\n\\nIncludes Lazy Evaluation for job expiration and Auto-Queue processing.")
inject_docstring("submit_cluster_job", "Submit a job to the cluster.\\n\\nUses an Energy-Aware 4-Pass Smart Scheduler to pack workloads efficiently.")
inject_docstring("reset_cluster_state", "Reset the entire cluster state, history, and savings to their initial default values.")
inject_docstring("inject_cluster_anomaly", "Inject a simulated thermal anomaly (85.0C) into a specified node for testing purposes.")
inject_docstring("predict_health", "Predict cluster health by scanning nodes for critical thermal thresholds (>= 80.0C).")
inject_docstring("migrate_and_remediate", "Autonomously isolate a critical node, extract its active jobs, and seamlessly migrate them to healthy nodes.")

with open('app.py', 'w') as f:
    f.write(content)
