from fastapi import FastAPI, HTTPException, status
from api.schemas import CustomerFeatures, PredictionResponse
from api.utils import load_model, format_features

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "A production-ready REST API using FastAPI to predict customer churn. "
        "The model uses RFM and customer-level behavioral features evaluated with XGBoost."
    ),
    version="1.0.0"
)

# Load the pre-trained model once at startup
try:
    model = load_model()
except Exception as e:
    model = None
    print(f"Warning: XGBoost model could not be loaded at startup. Error: {e}")

@app.get("/", summary="Root Message")
def read_root():
    """Returns a simple greeting message for the API."""
    return {"message": "Customer Churn Prediction API"}

@app.get("/health", summary="API Health Check")
def health_check():
    """Validates if the API is running and the pre-trained model is loaded."""
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "details": "Model file model.joblib could not be loaded."
        }
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Customer Churn"
)
def predict_churn(features: CustomerFeatures):
    """
    Predicts customer churn status based on behavioral features.
    
    Returns:
    - **prediction**: 0 (Active) or 1 (Churned)
    - **probability**: Probability score of churn (0.0 to 1.0)
    - **confidence_score**: Confidence level of the classification decision
    """
    global model
    if model is None:
        try:
            model = load_model()
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML model is currently unavailable on the server."
            )
            
    try:
        # Format the features dictionary into a structured dataframe
        input_data = format_features(features.dict())
        
        # Run inference
        pred = int(model.predict(input_data)[0])
        prob = float(model.predict_proba(input_data)[0][1])
        
        # Confidence score calculation
        conf = prob if pred == 1 else 1.0 - prob
        
        return PredictionResponse(
            prediction=pred,
            probability=prob,
            confidence_score=conf
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution error: {str(e)}"
        )
