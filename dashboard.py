import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# Import project modules
from data_simulation import generate_synthetic_cloud_data
from preprocessing import DataPreprocessor
from model_training import LSTMModel, RandomForestModel
from prediction import WorkloadPredictor
from optimization_engine import ResourceOptimizer, format_indian_currency
from autoscaler import AutoScaler
from cloud_simulator import CloudSimulator
from scenarios import WorkloadScenario

# Set page configuration
st.set_page_config(
    page_title="Cloud AI Resource Optimizer",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHED RESOURCES ---
@st.cache_resource
def load_data_and_model():
    """Load data and model only once."""
    # 1. Load Data
    data_path = "data/simulated_cloud_data.csv"
    if not os.path.exists(data_path):
        generate_synthetic_cloud_data(output_path=data_path)
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    # 3. Load Model (Prefer LSTM)
    model_type = 'lstm'
    model_path = "models/lstm_model.h5"
    if not os.path.exists(model_path):
        # Fallback to RF if LSTM missing
        model_type = 'rf'
        model_path = "models/rf_model.pkl"
    
    predictor = WorkloadPredictor(model_type=model_type, model_path=model_path)
    
    return df_clean, predictor, preprocessor

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    st.subheader("Workload Patterns")
    scenario_option = st.radio(
        "Select Traffic Scenario:",
        ["Baseline (Normal)", "Sudden Traffic Spike", "Night-Time Low Load", "Flash Sale Scenario", "Gradual Growth"]
    )
    
    st.divider()
    
    st.subheader("Simulation Parameters")
    intensity = st.slider("Workload Intensity (%)", min_value=50, max_value=150, value=100, step=10, 
                          help="Scale the overall traffic volume.")
    
    sla_threshold = st.slider("SLA Threshold (%)", min_value=60, max_value=90, value=75, step=5,
                              help="Utilization limit before scaling up.")
    
    vm_count_max = st.slider("Max Virtual Machines", min_value=1, max_value=20, value=10,
                             help="Maximum scaling limit.")
    
    st.divider()
    
    st.markdown("### 🤖 Strategy")
    ai_enabled = st.toggle("Enable AI Optimization", value=True)
    
    st.info("Toggle OFF to use Threshold-Based (Traditional) Scaling.")

# --- MAIN LOGIC ---

# Load resources
try:
    df_base, predictor, preprocessor = load_data_and_model()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# 1. Apply Scenario & Intensity
test_data = df_base.tail(200).copy().reset_index(drop=True) # last 200 simulation steps

if scenario_option != "Baseline (Normal)":
    test_data = WorkloadScenario.apply_scenario(test_data, scenario_option)

# Apply Intensity Multiplier
if intensity != 100:
    factor = intensity / 100.0
    test_data['cpu_usage'] = test_data['cpu_usage'] * factor
    test_data['ram_usage'] = test_data['ram_usage'] * factor
    test_data['requests'] = test_data['requests'] * factor

# Initialize Engines
optimizer = ResourceOptimizer()
autoscaler = AutoScaler(scale_up_threshold=sla_threshold/100.0, max_instances=vm_count_max)
simulator = CloudSimulator(optimizer=optimizer, autoscaler=autoscaler)

# 2. Run Simulation
# We run both baseline (fixed) AND the selected strategy (AI or Threshold) for comparison
baseline_results = simulator.simulate_baseline(test_data)

if ai_enabled:
    strategy_name = "AI-Optimized"
    sim_results = simulator.simulate_ai_optimized(
        test_data, predictor, preprocessor, lookback_window=24
    )
else:
    strategy_name = "Threshold-Based"
    sim_results = simulator.simulate_threshold_scaling(
        test_data, scale_up_threshold=sla_threshold/100.0
    )

# --- DASHBOARD UI ---

st.title("☁️ Cloud AI Resource Optimizer")
st.markdown(f"**Current Strategy:** `{strategy_name}` | **Scenario:** `{scenario_option}`")

# 3. KPI Metrics (Last Step / Aggregate)
col1, col2, col3, col4, col5 = st.columns(5)

current_cpu_util = sim_results['avg_cpu_utilization'] * 100
current_ram_util = sim_results['avg_ram_utilization'] * 100
total_cost = sim_results['total_cost']
cost_saving = baseline_results['total_cost'] - total_cost
sla_violation_rate = sim_results.get('sla_violation_rate', 0)
baseline_sla_rate = baseline_results.get('sla_violation_rate', 0)

with col1:
    st.metric("Avg CPU Utilization", f"{current_cpu_util:.1f}%", delta=f"{current_cpu_util - (baseline_results['avg_cpu_utilization']*100):.1f}%")
with col2:
    st.metric("Avg RAM Utilization", f"{current_ram_util:.1f}%", delta=f"{current_ram_util - (baseline_results['avg_ram_utilization']*100):.1f}%")
with col3:
    st.metric("SLA Violation Rate", f"{sla_violation_rate:.2f}%", delta=f"{sla_violation_rate - baseline_sla_rate:.2f}%", delta_color="inverse")
with col4:
    formatted_cost = format_indian_currency(total_cost)
    st.metric("Total Projected Cost", f"₹ {formatted_cost}", delta="- Saving" if cost_saving > 0 else "Loss", delta_color="inverse")
with col5:
    st.metric("Efficiency Score", f"{sim_results['avg_efficiency']:.3f}")


# 4. Graphs
tab1, tab2, tab3 = st.tabs(["📈 Utilization Analysis", "💰 Cost Comparison", "🧠 Explainability"])

with tab1:
    st.subheader("Resource Utilization Over Time")
    
    # Plotly Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("CPU Utilization", "RAM Utilization"))
    
    x_axis = test_data['timestamp'][-len(sim_results['cpu_utilizations']):]
    
    # CPU
    fig.add_trace(go.Scatter(x=x_axis, y=[u*100 for u in sim_results['cpu_utilizations']], name=f"{strategy_name} CPU", line=dict(color='#00CC96')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=[u*100 for u in baseline_results['cpu_utilizations']], name="Baseline CPU", line=dict(color='#EF553B', dash='dot')), row=1, col=1)
    
    # RAM
    fig.add_trace(go.Scatter(x=x_axis, y=[u*100 for u in sim_results['ram_utilizations']], name=f"{strategy_name} RAM", line=dict(color='#636EFA')), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=[u*100 for u in baseline_results['ram_utilizations']], name="Baseline RAM", line=dict(color='#FFA15A', dash='dot')), row=2, col=1)
    
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Cost Effectiveness")
    
    costs = [baseline_results['total_cost'], sim_results['total_cost']]
    names = ['Fixed Allocation (Baseline)', strategy_name]
    colors = ['#EF553B', '#00CC96']
    
    fig_cost = go.Figure(data=[
        go.Bar(name='Cost', x=names, y=costs, marker_color=colors, text=[f"₹{format_indian_currency(c)}" for c in costs], textposition='auto')
    ])
    fig_cost.update_layout(title_text='Total Simulated Cost (INR)')
    st.plotly_chart(fig_cost, use_container_width=True)
    
    if cost_saving > 0:
        st.success(f"**Savings:** ₹ {format_indian_currency(cost_saving)} compared to fixed allocation.")

with tab3:
    st.subheader("Decision Logic & Explainability")
    
    # Get last decision logic
    if ai_enabled and 'allocations' in sim_results:
        # Example logic extraction
        last_util = sim_results['cpu_utilizations'][-1]
        
        st.markdown(f"""
        ### Why was this decision made?
        
        The **AI-Optimized Strategy** uses an LSTM neural network to predict workload 24 steps ahead.
        
        - **Workload Analysis**: The model detected a CPU utilization pattern of **{current_cpu_util:.1f}%**.
        - **Scaling Action**: Based on the SLA threshold of **{sla_threshold}%**, the system decided to maintain optimal instance count.
        - **Vertical Scaling**: Resource allocation was dynamically adjusted to match the predicted curve, minimizing waste.
        """)
        
    else:
        st.markdown(f"""
        ### Threshold Logic
        
        The **Traditional Strategy** uses simple reactive rules:
        
        - **Scale Up**: If utilization > **{sla_threshold}%**
        - **Scale Down**: If utilization < **40%**
        
        *Currently, this strategy reacts only after the threshold is breached, often leading to lag.*
        """)

# Footer
st.markdown("---")
st.caption("Cloud AI Resource Optimization Project | Dashboard v1.0")
