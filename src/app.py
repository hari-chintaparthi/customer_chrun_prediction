import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# Add current dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from src import config
importlib.reload(config)

try:
    from src.train import train_pipeline, MLFLOW_AVAILABLE
except Exception as e:
    train_pipeline = None
    MLFLOW_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="ChurnGuard AI | Enterprise Churn Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Premium Glassmorphic Dark Theme Vibes)
st.markdown("""
<style>
    /* Gradient Background for Header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .main-header h1 {
        color: #f8fafc !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.05em;
        margin: 0;
        font-size: 2.8rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Risk Levels Cards */
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
        margin-bottom: 1rem;
        text-align: center;
    }
    .risk-low {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
    }
    .risk-medium {
        background: rgba(245, 158, 11, 0.15);
        border-color: rgba(245, 158, 11, 0.4);
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.15);
        border-color: rgba(239, 68, 68, 0.4);
    }
    
    /* Stats Cards */
    .stat-container {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stat-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load all models
@st.cache_resource(show_spinner=False)
def load_all_models():
    models = {}
    for name in ["Logistic_Regression", "Random_Forest", "LightGBM", "XGBoost"]:
        path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models

# Sidebar Content
st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>🛡️ ChurnGuard AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94a3b8;'>Enterprise Customer Churn Suite</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

all_models = load_all_models()
if all_models:
    st.sidebar.markdown("### 🔮 Model Selection")
    model_options = list(all_models.keys())
    # Recommend XGBoost as default
    default_idx = model_options.index("XGBoost") if "XGBoost" in model_options else 0
    
    selected_model_name = st.sidebar.selectbox(
        "Select ML Model",
        options=model_options,
        index=default_idx,
        help="Choose which model to use for predictions. XGBoost is recommended as the Champion."
    )
    model = all_models[selected_model_name]
    st.sidebar.success(f"💡 **Active**: {selected_model_name}")
    st.sidebar.info("🏆 **XGBoost** is recommended based on F1-Score (0.86) and Recall (0.92) metrics.")
else:
    model = None
    st.sidebar.warning("⚠️ No models found. Please train models in the MLflow tab first.")

# Load data for EDA and visualizations
@st.cache_data(show_spinner="Loading dataset overview...")
def load_eda_data():
    if os.path.exists(config.ORIGINAL_DATASET_PATH):
        # Read a subset to keep app fast
        return pd.read_csv(config.ORIGINAL_DATASET_PATH)
    return None

df_data = load_eda_data()

# Header banner
st.markdown("""
    <div class="main-header">
        <h1>ChurnGuard AI Platform</h1>
        <p>Real-time customer churn intelligence engineered with SMOTE, XGBoost, and MLflow tracking.</p>
    </div>
""", unsafe_allow_html=True)

# Tabs
tab_overview, tab_single, tab_batch, tab_mlflow = st.tabs([
    "📊 Dataset & Visualizations", 
    "👤 Single Customer Predictor", 
    "📂 Batch File Predictor", 
    "🔬 MLflow Experimentation"
])

# ================= TAB 1: OVERVIEW & EDA =================
with tab_overview:
    st.subheader("Exploratory Data Analysis & Feature Insights")
    
    if df_data is None:
        st.warning(f"Original dataset not found at `{config.ORIGINAL_DATASET_PATH}`. Please check path or verify dataset exists.")
    else:
        # Key Dataset Stats
        total_records = len(df_data)
        churn_records = df_data[config.TARGET].sum()
        normal_records = total_records - churn_records
        churn_rate = (churn_records / total_records) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="stat-container"><div class="stat-val">{total_records:,}</div><div class="stat-label">Total Profiles</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-container"><div class="stat-val">{normal_records:,}</div><div class="stat-label">Active (Class 0)</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-container"><div class="stat-val" style="color: #f43f5e;">{churn_records:,}</div><div class="stat-label">Churned (Class 1)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="stat-container"><div class="stat-val" style="color: #fb7185;">{churn_rate:.3f}%</div><div class="stat-label">Churn Rate (Imbalance)</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Grid of Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### ⚖️ Class Imbalance Visualization")
            fig_imbalance = px.bar(
                x=["Active Profiles (0)", "Churned Profiles (1)"],
                y=[normal_records, churn_records],
                labels={'x': 'Status', 'y': 'Count'},
                color=["Active", "Churned"],
                color_discrete_map={"Active": "#10b981", "Churned": "#ef4444"},
                template="plotly_dark",
                height=350
            )
            fig_imbalance.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_imbalance, width="stretch")
            st.caption("Notice the extreme class imbalance (only 0.17% churn cases). This makes SMOTE an absolute necessity for model training.")
            
        with col_chart2:
            st.markdown("##### 💳 Distribution of Average Monthly Spending (V29)")
            fig_dist = px.histogram(
                df_data,
                x="V29",
                color="Target",
                color_discrete_map={0: "#10b981", 1: "#ef4444"},
                nbins=50,
                log_y=True,
                labels={"V29": "Average Monthly Spending ($)", "Target": "Churn Status"},
                template="plotly_dark",
                height=350
            )
            fig_dist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_dist, width="stretch")
            st.caption("Distribution of average monthly spending (V29) on log scale. Green indicates active customers, Red indicates churned.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Heatmap of key correlations
        col_heat, col_imp = st.columns([3, 2])
        
        with col_heat:
            st.markdown("##### 🔗 Feature Correlation Heatmap (Selected Features)")
            # Select config.FEATURES and Target to visualize correlation
            corr_features = config.FEATURES + ["Target"]
            corr_matrix = df_data[corr_features].corr()
            
            fig_heat = px.imshow(
                corr_matrix,
                labels=dict(x="Features", y="Features", color="Correlation"),
                x=corr_features,
                y=corr_features,
                color_continuous_scale="RdBu",
                template="plotly_dark",
                height=400
            )
            fig_heat.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_heat, width="stretch")
            st.caption("Correlation matrix for the first 10 engineered features and the Target.")
            
        with col_imp:
            st.markdown("##### 🚀 XGBoost Feature Importances")
            if model is not None:
                try:
                    importances = model.feature_importances_
                    feat_names = [config.FEATURE_MAP.get(f, f) for f in config.FEATURES]
                    df_imp = pd.DataFrame({
                        "Feature": feat_names,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False).head(10)
                    
                    fig_imp = px.bar(
                        df_imp,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale="Viridis",
                        template="plotly_dark",
                        height=400
                    )
                    fig_imp.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_imp, width="stretch")
                    st.caption("Top 10 features driving churn prediction according to the champion XGBoost model.")
                except Exception as e:
                    st.info("Feature importance will be available after loading/training a valid tree-based champion model.")
            else:
                st.info("XGBoost model is not loaded. Train a model to see feature importances.")

# ================= TAB 2: SINGLE CUSTOMER PREDICTOR =================
with tab_single:
    st.subheader("Individual Customer Profile Churn Analysis")
    
    if model is None:
        st.error("Champion model is not loaded. Please go to the MLflow tab to train models first.")
    else:
        st.markdown("Configure the customer's behavioral and transaction indicators below to generate an instantaneous churn risk assessment:")
        
        # Dynamic column layout for top features with realistic scales
        col_input1, col_input2 = st.columns(2)
        input_values = {}
        
        for i, feat in enumerate(config.FEATURES):
            friendly_name = config.FEATURE_MAP.get(feat, feat)
            label = f"{feat} - {friendly_name}"
            
            target_col = col_input1 if i % 2 == 0 else col_input2
            
            # Retrieve realistic configs or fallback
            cfg = config.REALISTIC_CONFIG.get(feat, {
                "min": -10.0,
                "max": 10.0,
                "default": 0.0,
                "step": 0.1,
                "format": "%.2f"
            })
            
            with target_col:
                input_values[feat] = st.number_input(
                    label,
                    value=float(cfg["default"]),
                    step=float(cfg["step"]),
                    format=cfg["format"],
                    key=f"input_{feat}"
                )
            
        # Convert to z-scores under the hood for model prediction
        scaled_input_values = {}
        for feat in config.FEATURES:
            val = input_values[feat]
            cfg = config.REALISTIC_CONFIG.get(feat, {})
            mean = cfg.get("mean", 0.0)
            std = cfg.get("std", 1.0)
            scaled_input_values[feat] = (val - mean) / std
            
        # Perform live prediction using scaled features
        input_df = pd.DataFrame([scaled_input_values])
        input_df = input_df[config.FEATURES]
        
        prediction_success = False
        try:
            # Check model compatibility
            if hasattr(model, "n_features_in_"):
                expected_features = model.n_features_in_
            else:
                expected_features = len(model.feature_importances_)
                
            if expected_features != len(config.FEATURES):
                st.error(f"⚠️ Model Feature Mismatch: Trained model expects {expected_features} features, but the configuration has {len(config.FEATURES)}. Please go to the 'MLflow Experimentation' tab and retrain the model.")
            else:
                churn_prob = model.predict_proba(input_df)[0][1]
                churn_pred = model.predict(input_df)[0]
                prediction_success = True
        except Exception as e:
            st.error(f"Prediction failed: {e}. Please ensure the trained model matches the current feature configuration by retraining in the MLflow tab.")
        
        if prediction_success:
            st.markdown("---")
            st.markdown("### Churn Analysis Output")
            
            col_out1, col_out2 = st.columns([1, 2])
            
            with col_out1:
                # Gauge Chart or colored box
                risk_pct = churn_prob * 100
                
                if risk_pct < 30:
                    risk_class = "LOW RISK"
                    risk_style = "risk-low"
                    advice = "Customer shows stable account usage. Normal relationship management is recommended."
                elif risk_pct < 70:
                    risk_class = "MEDIUM RISK"
                    risk_style = "risk-medium"
                    advice = "Customer displays warning signs of churn. Formulate targeted engagement campaigns."
                else:
                    risk_class = "HIGH RISK"
                    risk_style = "risk-high"
                    advice = "Customer has a high probability of churn. Immediate proactive retention efforts required!"
                    
                st.markdown(f"""
                    <div class="risk-card {risk_style}">
                        <h3 style="margin-top: 0; margin-bottom: 0.5rem; color: #ffffff;">{risk_class}</h3>
                        <div style="font-size: 3.5rem; font-weight: 800; line-height: 1;">{risk_pct:.1f}%</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem; color: rgba(255, 255, 255, 0.85);">Churn Probability</div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Advisor Verdict:** {advice}")
                
            with col_out2:
                st.markdown("##### 🛠️ Tailored Customer Retention Recommendations")
                
                recommendations = []
                
                # Check metrics against realistic thresholds
                if input_values.get("V4", 0.0) > 3.0:
                    recommendations.append("🚨 **Customer Support Escalation**: Customer support tickets opened count is elevated. Schedule a priority check-in call with an Account Manager to resolve ongoing disputes or issues.")
                
                if input_values.get("V10", 25.0) < 10.0:
                    recommendations.append("💳 **Limit Utilization Push**: Limit utilization is abnormally low. Send a targeted credit card promotional offer (cashback/low-APR) to stimulate card usage.")
                    
                if input_values.get("V12", 0.0) > 3.0:
                    recommendations.append("📱 **Technical Support Outreach**: High frequency of application crashes. Trigger automated software diagnostic checks or contact client with app-reinstallation/troubleshooting assistance.")
                    
                # V24 and V26 are not in current feature configuration, so they default to neutral states
                if input_values.get("V24", 0.0) > 1.0:
                    recommendations.append("⚠️ **Late Payment Grace Offer**: Customer has late payment violations. Offer a one-time fee waiver/grace period for payment along with payment reminder options.")
                    
                if input_values.get("V26", 0.0) < -1.0:
                    recommendations.append("🎁 **Loyalty Incentive**: Low referral invites. Offer double reward points on grocery and restaurant purchases to rebuild loyalty connection.")
                    
                if not recommendations:
                    recommendations.append("✅ **Healthy Account State**: Customer behavioral metrics are in equilibrium. Keep enrolling the customer in normal seasonal offers.")
                    
                for rec in recommendations:
                    st.write(rec)
                    
                # Plot contributions based on z-score deviation from mean
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 📌 Key Drivers of this Profile's Risk Score")
                
                # Extract feature importance dynamically (supports tree-based and linear models)
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_[0])
                    if importances.sum() > 0:
                        importances = importances / importances.sum()
                else:
                    importances = np.ones(len(config.FEATURES)) / len(config.FEATURES)
                    
                drivers = []
                for i, feat in enumerate(config.FEATURES):
                    val = scaled_input_values[feat]
                    drivers.append({
                        "Feature": f"{feat} - {config.FEATURE_MAP.get(feat, feat)}",
                        "Value": val,
                        "Impact Score": abs(val) * importances[i]
                    })
                df_drivers = pd.DataFrame(drivers).sort_values(by="Impact Score", ascending=False).head(4)
                
                fig_drivers = px.bar(
                    df_drivers,
                    x="Impact Score",
                    y="Feature",
                    orientation="h",
                    color="Value",
                    color_continuous_scale="rdbu",
                    template="plotly_dark",
                    height=220
                )
                fig_drivers.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_drivers, width="stretch")

# ================= TAB 3: BATCH FILE PREDICTOR =================
with tab_batch:
    st.subheader("Batch Customer Profile Churn Upload")
    
    if model is None:
        st.error("Champion model is not loaded. Please go to the MLflow tab to train models first.")
    else:
        st.write("Upload a CSV file containing active customer profiles (with columns matching `V1` to `V29`). The platform will batch process the profiles, score their churn probabilities, classify them into risk tiers, and generate a downloadable report.")
        
        # Provide sample download
        if os.path.exists(config.SUBSET_DATASET_PATH):
            with open(config.SUBSET_DATASET_PATH, "r") as f:
                csv_data = f.read()
            st.download_button(
                label="📥 Download Sample Batch CSV File",
                data=csv_data,
                file_name="sample_customer_profiles.csv",
                mime="text/csv"
            )
            
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                
                # Check features are in the file
                missing_features = [f for f in config.FEATURES if f not in batch_df.columns]
                if missing_features:
                    st.error(f"Uploaded CSV is missing required features: {missing_features}. Please download and inspect the sample batch CSV above.")
                else:
                    st.success("CSV file successfully loaded!")
                    
                    # Run batch predictions
                    with st.spinner("Processing batch customer profiles..."):
                        # Extract features in correct order
                        X_batch = batch_df[config.FEATURES]
                        
                        # Predict probability
                        probs = model.predict_proba(X_batch)[:, 1]
                        preds = model.predict(X_batch)
                        
                        # Add results to dataframe
                        result_df = batch_df.copy()
                        result_df["Churn_Probability"] = probs
                        result_df["Predicted_Churn"] = preds
                        result_df["Risk_Level"] = pd.cut(
                            result_df["Churn_Probability"],
                            bins=[-0.1, 0.3, 0.7, 1.01],
                            labels=["Low Risk", "Medium Risk", "High Risk"]
                        )
                        
                        # Sort by highest probability first
                        result_df = result_df.sort_values(by="Churn_Probability", ascending=False)
                        
                    # Summary metrics
                    total_uploaded = len(result_df)
                    predicted_churn_count = (result_df["Predicted_Churn"] == 1).sum()
                    overall_risk_rate = (predicted_churn_count / total_uploaded) * 100
                    avg_prob = result_df["Churn_Probability"].mean() * 100
                    
                    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                    with col_b1:
                        st.markdown(f'<div class="stat-container"><div class="stat-val">{total_uploaded:,}</div><div class="stat-label">Profiles Processed</div></div>', unsafe_allow_html=True)
                    with col_b2:
                        st.markdown(f'<div class="stat-container"><div class="stat-val" style="color: #ef4444;">{predicted_churn_count:,}</div><div class="stat-label">Predicted Churns</div></div>', unsafe_allow_html=True)
                    with col_b3:
                        st.markdown(f'<div class="stat-container"><div class="stat-val" style="color: #fb7185;">{overall_risk_rate:.2f}%</div><div class="stat-label">Predicted Churn Rate</div></div>', unsafe_allow_html=True)
                    with col_b4:
                        st.markdown(f'<div class="stat-container"><div class="stat-val" style="color: #38bdf8;">{avg_prob:.2f}%</div><div class="stat-label">Average Churn Risk</div></div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Download Results
                    csv_output = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Scored Batch Predictions CSV",
                        data=csv_output,
                        file_name=f"scored_churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Display table
                    st.markdown("##### 🔍 Scored Customers Data View (Top 100 Highest Risk)")
                    display_cols = ["Churn_Probability", "Risk_Level", "Predicted_Churn"] + config.FEATURES[:5] + ["V29"]
                    st.dataframe(
                        result_df[display_cols].head(100).style.format({
                            "Churn_Probability": "{:.2%}",
                            "V29": "{:.2f}"
                        }),
                        width="stretch"
                    )
            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")

# ================= TAB 4: MLFLOW RUN TRACKER & RETRAIN =================
with tab_mlflow:
    st.subheader("MLflow Experiment Tracking & Training Orchestrator")
    
    # MLflow Setup Status
    if MLFLOW_AVAILABLE:
        try:
            import mlflow
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        except Exception as e:
            MLFLOW_AVAILABLE = False
            
    if not MLFLOW_AVAILABLE:
        st.warning("⚠️ **MLflow Experiment Tracking is currently offline or unavailable** (e.g., due to package conflicts on Streamlit Cloud Python 3.14).")
        st.info("The application has gracefully bypassed MLflow. Single predictions, batch predictions, and data visualizations are fully functional using the pre-trained champion model. You can still trigger local retraining below, but runs will not be logged to MLflow.")
    
    col_t1, col_t2 = st.columns([2, 3])
    
    with col_t1:
        st.markdown("##### ⚙️ Training Execution Panel")
        st.write("Trigger the complete end-to-end training pipeline. The script will:")
        st.markdown("""
        1. Load dataset from file system
        2. Apply SMOTE to training split
        3. Train Logistic Regression, Random Forest, LightGBM, and XGBoost
        4. Log hyperparameters, F1, Recall, and ROC-AUC in MLflow (if available)
        5. Save XGBoost as champion model
        """)
        
        # Check if local dataset exists
        dataset_exists = os.path.exists(config.ORIGINAL_DATASET_PATH)
        can_train = dataset_exists and train_pipeline is not None
        
        if not dataset_exists:
            st.error(f"Dataset NOT found at `{config.ORIGINAL_DATASET_PATH}`. Training cannot be triggered.")
        elif train_pipeline is None:
            st.error("Training module is unavailable.")
        else:
            st.success("Creditcard.csv dataset verified. Ready to train.")
            
        retrain_btn = st.button("🚀 Trigger Model Re-training Pipeline", disabled=not can_train)
        
        if retrain_btn and train_pipeline is not None:
            log_area = st.empty()
            progress_bar = st.progress(0)
            
            logs = []
            def update_log(msg):
                logs.append(msg)
                log_area.code("\n".join(logs), language="text")
                # update progress bar based on message keywords
                if "Loading dataset" in msg:
                    progress_bar.progress(10)
                elif "Splitting data" in msg:
                    progress_bar.progress(25)
                elif "Applying SMOTE" in msg:
                    progress_bar.progress(40)
                elif "Training Logistic_Regression" in msg:
                    progress_bar.progress(50)
                elif "Training Random_Forest" in msg:
                    progress_bar.progress(65)
                elif "Training LightGBM" in msg:
                    progress_bar.progress(75)
                elif "Training XGBoost" in msg:
                    progress_bar.progress(85)
                elif "Saved Champion" in msg:
                    progress_bar.progress(95)
                elif "Pipeline completed" in msg:
                    progress_bar.progress(100)
                    
            try:
                with st.spinner("Executing pipeline..."):
                    performances = train_pipeline(progress_callback=update_log)
                st.success("Model pipeline run successfully! Reloading models...")
                # Force refresh page
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.code(str(e))
                
    with col_t2:
        st.markdown("##### 🔬 MLflow Run Logs (Experiment: Customer_Churn_Prediction)")
        
        if not MLFLOW_AVAILABLE:
            st.info("Run history and comparisons are not available because MLflow is disabled in this environment.")
        else:
            # Load run history from MLflow local mlruns
            try:
                runs_df = mlflow.search_runs(experiment_names=["Customer_Churn_Prediction"])
                if runs_df is not None and not runs_df.empty:
                    # Format runs dataframe
                    runs_display = runs_df.copy()
                    if "tags.mlflow.runName" in runs_display.columns:
                        runs_display["run_name"] = runs_display["tags.mlflow.runName"]
                    
                    # Keep important columns
                    keep_cols = [
                        "run_name", 
                        "status", 
                        "metrics.recall", 
                        "metrics.f1_score", 
                        "metrics.roc_auc", 
                        "start_time"
                    ]
                    
                    # Verify columns exist
                    keep_cols = [c for c in keep_cols if c in runs_display.columns]
                    runs_display = runs_display[keep_cols]
                    
                    # Rename columns for presentation
                    rename_map = {
                        "run_name": "Model / Run Name",
                        "status": "Status",
                        "metrics.recall": "Recall",
                        "metrics.f1_score": "F1-Score",
                        "metrics.roc_auc": "ROC-AUC",
                        "start_time": "Date Run"
                    }
                    runs_display = runs_display.rename(columns=rename_map)
                    
                    # Sort by Date Run descending
                    if "Date Run" in runs_display.columns:
                        runs_display = runs_display.sort_values(by="Date Run", ascending=False)
                        # Convert date format
                        runs_display["Date Run"] = pd.to_datetime(runs_display["Date Run"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    st.dataframe(
                        runs_display.style.format({
                            "Recall": "{:.4f}",
                            "F1-Score": "{:.4f}",
                            "ROC-AUC": "{:.4f}"
                        }),
                        width="stretch"
                    )
                    
                    # Compare models Chart
                    st.markdown("##### 📊 Model Performance Comparison")
                    # Group by Model Name and get average metrics
                    runs_grouped = runs_display.groupby("Model / Run Name")[["Recall", "F1-Score", "ROC-AUC"]].mean().reset_index()
                    
                    # Plotly grouped bar chart
                    fig_comp = go.Figure()
                    for metric in ["Recall", "F1-Score", "ROC-AUC"]:
                        fig_comp.add_trace(go.Bar(
                            x=runs_grouped["Model / Run Name"],
                            y=runs_grouped[metric],
                            name=metric,
                            text=[f"{v:.3f}" for v in runs_grouped[metric]],
                            textposition='auto'
                        ))
                    
                    fig_comp.update_layout(
                        barmode='group',
                        template="plotly_dark",
                        height=350,
                        margin=dict(l=20, r=20, t=30, b=20),
                        yaxis=dict(range=[0, 1.05])
                    )
                    st.plotly_chart(fig_comp, width="stretch")
                    
                else:
                    st.info("No runs found in local MLflow repository. Trigger the model re-training pipeline to register runs.")
            except Exception as e:
                st.info("No run logs available yet. Please execute the training pipeline to generate run history.")
            # Debug detail
            # st.error(str(e))
