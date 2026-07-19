import os
import joblib
import pandas as pd
from xgboost import XGBClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

def load_model() -> XGBClassifier:
    """Load the pre-trained XGBoost model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure training has been executed.")
    return joblib.load(MODEL_PATH)

def format_features(features_dict: dict) -> pd.DataFrame:
    """Format the input dictionary into a pandas DataFrame matching XGBoost feature order."""
    feature_order = [
        "Recency", "Frequency", "Monetary", "Average_Order_Value",
        "Average_Basket_Size", "Unique_Products", "Customer_Lifetime",
        "Purchase_Interval", "Transactions_Per_Month", "Product_Diversity"
    ]
    
    mapping = {
        "Recency": "Recency",
        "Frequency": "Frequency",
        "Monetary": "Monetary",
        "AverageOrderValue": "Average_Order_Value",
        "AverageBasketSize": "Average_Basket_Size",
        "UniqueProducts": "Unique_Products",
        "CustomerLifetime": "Customer_Lifetime",
        "PurchaseInterval": "Purchase_Interval",
        "TransactionsPerMonth": "Transactions_Per_Month",
        "ProductDiversity": "Product_Diversity"
    }
    
    mapped_features = {mapping[k]: [v] for k, v in features_dict.items() if k in mapping}
    return pd.DataFrame(mapped_features)[feature_order]
