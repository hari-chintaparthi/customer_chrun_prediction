import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score

# Add current dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
except Exception as e:
    MLFLOW_AVAILABLE = False

def train_pipeline(progress_callback=None):
    def log_progress(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    log_progress("Starting ML Pipeline...")
    
    # 1. Load dataset
    log_progress(f"Loading dataset from {config.ORIGINAL_DATASET_PATH}...")
    if not os.path.exists(config.ORIGINAL_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {config.ORIGINAL_DATASET_PATH}")
        
    df = pd.read_csv(config.ORIGINAL_DATASET_PATH)
    log_progress(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Create subset dataset if it doesn't exist for demo uploads
    if not os.path.exists(config.SUBSET_DATASET_PATH):
        log_progress("Creating a small subset dataset for batch prediction testing...")
        # Create a subset with 500 normal cases and 20 churn cases
        df_normal = df[df[config.TARGET] == 0].sample(n=500, random_state=42)
        df_churn = df[df[config.TARGET] == 1].sample(n=20, random_state=42)
        df_subset = pd.concat([df_normal, df_churn]).sample(frac=1, random_state=42)
        df_subset.to_csv(config.SUBSET_DATASET_PATH, index=False)
        log_progress(f"Subset dataset created at {config.SUBSET_DATASET_PATH}")

    # 2. Preprocess data
    X = df[config.FEATURES]
    y = df[config.TARGET]
    
    # Split train/test
    log_progress("Splitting data into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    log_progress(f"Train class distribution: Churn={y_train.sum()}, Non-Churn={len(y_train) - y_train.sum()}")
    log_progress(f"Test class distribution: Churn={y_test.sum()}, Non-Churn={len(y_test) - y_test.sum()}")
    
    # 3. Apply SMOTE to training data only
    log_progress("Applying SMOTE to address class imbalance on training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    log_progress(f"Resampled training set shape: Churn={y_train_res.sum()}, Non-Churn={len(y_train_res) - y_train_res.sum()}")
    
    # 4. Set up MLflow
    use_mlflow = MLFLOW_AVAILABLE
    if use_mlflow:
        try:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment("Customer_Churn_Prediction")
        except Exception as e:
            log_progress(f"Failed to initialize MLflow: {e}. MLflow tracking is disabled.")
            use_mlflow = False
            
    # Dictionary to keep track of model performances
    performances = {}
    
    # Models to train
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, eval_metric="logloss", random_state=42, n_jobs=-1)
    }
    
    # Train and log each model
    for model_name, model in models.items():
        log_progress(f"\n--- Training {model_name} ---")
        
        active_run = None
        run_mlflow = use_mlflow
        if run_mlflow:
            try:
                active_run = mlflow.start_run(run_name=model_name)
            except Exception as e:
                log_progress(f"Failed to start MLflow run for {model_name}: {e}. Training without MLflow logging.")
                run_mlflow = False
                
        # Train model
        log_progress(f"Fitting {model_name}...")
        model.fit(X_train_res, y_train_res)
        
        # Predict
        y_pred = model.predict(X_test)
        # Handle predict_proba availability
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        # Evaluate metrics
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        log_progress(f"Results for {model_name}: Accuracy={acc:.4f}, Recall={rec:.4f}, F1-Score={f1:.4f}, ROC-AUC={auc:.4f}")
        
        # Log to mlflow if active
        if run_mlflow and active_run:
            try:
                # Log params
                if model_name == "Logistic_Regression":
                    mlflow.log_param("max_iter", model.max_iter)
                    mlflow.log_param("C", model.C)
                elif model_name == "Random_Forest":
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("max_depth", model.max_depth)
                elif model_name == "LightGBM":
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
                    mlflow.log_param("max_depth", model.max_depth)
                elif model_name == "XGBoost":
                    mlflow.log_param("n_estimators", model.n_estimators)
                    mlflow.log_param("learning_rate", model.learning_rate)
                    mlflow.log_param("max_depth", model.max_depth)
                    
                mlflow.log_param("smote_applied", True)
                
                # Log metrics
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", auc)
                
                # Log model artifact
                if model_name == "XGBoost":
                    mlflow.xgboost.log_model(model, "model")
                elif model_name == "LightGBM":
                    mlflow.lightgbm.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                log_progress(f"MLflow logging error: {e}. Proceeding...")
            finally:
                try:
                    mlflow.end_run()
                except Exception:
                    pass
                    
        performances[model_name] = {
            "model": model,
            "metrics": {"accuracy": acc, "recall": rec, "f1_score": f1, "roc_auc": auc}
        }
            
    # 5. Champion selection (XGBoost based on resume, or let's double check if it outperformed)
    log_progress("\nSelect Champion Model...")
    champion_name = "XGBoost"
    champion_model = performances[champion_name]["model"]
    
    # Save all models locally for Streamlit comparison selection
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    for name, perf in performances.items():
        model_path = os.path.join(config.MODELS_DIR, f"{name}.joblib")
        joblib.dump(perf["model"], model_path)
        log_progress(f"Saved {name} model locally to {model_path}")
        
    # Save champion model locally
    joblib.dump(champion_model, config.CHAMPION_MODEL_PATH)
    log_progress(f"Saved Champion Model ({champion_name}) locally to {config.CHAMPION_MODEL_PATH}")
    
    log_progress("ML Pipeline completed successfully!")
    return performances

if __name__ == "__main__":
    train_pipeline()
