import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.replace(os.sep, '/')}"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)


# Datasets
ORIGINAL_DATASET_PATH = r"C:\Users\C Hari\Downloads\creditcard.csv"
SUBSET_DATASET_PATH = os.path.join(DATA_DIR, "creditcard_subset.csv")
CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, "champion_xgb.joblib")

# Map of raw feature names (V1 to V29) to user-friendly business/behavioral descriptions
FEATURE_MAP = {
    "V1": "Account Age (Months)",
    "V2": "Avg Transaction Frequency (Weekly)",
    "V3": "Avg Daily Screen Time (Mins)",
    "V4": "Customer Support Tickets Opened",
    "V5": "Last Transaction Amount ($)",
    "V6": "Account Login Frequency (Daily)",
    "V7": "Profile Completion %",
    "V8": "Auto-Pay Enrolled (0=No, 1=Yes)",
    "V9": "Direct Deposit Active (0=No, 1=Yes)",
    "V10": "Credit Card Limit Utilization %",
    "V11": "Monthly Service Fees ($)",
    "V12": "App Crashes (Last 30 Days)",
    "V13": "Age of Customer (Years)",
    "V14": "Total Account Balance ($)",
    "V15": "Average Cash Back Earned ($)",
    "V16": "E-Statements Active (0=No, 1=Yes)",
    "V17": "Promo Codes Applied (Last 6m)",
    "V18": "Card Swipes - Grocery Stores",
    "V19": "Card Swipes - Restaurants",
    "V20": "Card Swipes - Travel/Flights",
    "V21": "Overseas Transaction Count",
    "V22": "Disputed Transactions Count",
    "V23": "Avg Days Between Logins",
    "V24": "Late Payment Violations Count",
    "V25": "Customer Loyalty Tier Score",
    "V26": "Referral Invites Sent",
    "V27": "Credit Score Tier (1-5)",
    "V28": "SMS Notifications Clicked",
    "V29": "Avg Monthly Spending ($)",
}

# Reverse map for convenience
REVERSE_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# Ordered list of selected features (top 10 based on correlation analysis)
FEATURES = ["V17", "V14", "V12", "V10", "V3", "V16", "V7", "V11", "V4", "V18"]
TARGET = "Target"

# Configuration for converting realistic business metrics in the UI to standardized model inputs
REALISTIC_CONFIG = {
    "V3": {
        "min": 0.0,
        "max": 480.0,
        "default": 60.0,
        "mean": 60.0,
        "std": 30.0,
        "step": 5.0,
        "format": "%.1f"
    },
    "V4": {
        "min": 0.0,
        "max": 20.0,
        "default": 1.0,
        "mean": 1.5,
        "std": 1.5,
        "step": 1.0,
        "format": "%.0f"
    },
    "V7": {
        "min": 0.0,
        "max": 100.0,
        "default": 85.0,
        "mean": 80.0,
        "std": 15.0,
        "step": 1.0,
        "format": "%.0f"
    },
    "V10": {
        "min": 0.0,
        "max": 100.0,
        "default": 25.0,
        "mean": 30.0,
        "std": 20.0,
        "step": 1.0,
        "format": "%.0f"
    },
    "V11": {
        "min": 0.0,
        "max": 200.0,
        "default": 15.0,
        "mean": 20.0,
        "std": 15.0,
        "step": 1.0,
        "format": "%.2f"
    },
    "V12": {
        "min": 0.0,
        "max": 20.0,
        "default": 0.0,
        "mean": 1.0,
        "std": 1.5,
        "step": 1.0,
        "format": "%.0f"
    },
    "V14": {
        "min": 0.0,
        "max": 100000.0,
        "default": 5000.0,
        "mean": 5000.0,
        "std": 4000.0,
        "step": 100.0,
        "format": "%.2f"
    },
    "V16": {
        "min": 0.0,
        "max": 1.0,
        "default": 1.0,
        "mean": 0.5,
        "std": 0.5,
        "step": 1.0,
        "format": "%.0f"
    },
    "V17": {
        "min": 0.0,
        "max": 20.0,
        "default": 2.0,
        "mean": 2.0,
        "std": 2.0,
        "step": 1.0,
        "format": "%.0f"
    },
    "V18": {
        "min": 0.0,
        "max": 100.0,
        "default": 8.0,
        "mean": 10.0,
        "std": 8.0,
        "step": 1.0,
        "format": "%.0f"
    }
}
