from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    Recency: float = Field(..., description="Days since last purchase", json_schema_extra={"example": 45.0})
    Frequency: int = Field(..., description="Number of unique invoices", json_schema_extra={"example": 5})
    Monetary: float = Field(..., description="Total money spent", json_schema_extra={"example": 1250.50})
    AverageOrderValue: float = Field(..., description="Average spend per transaction", json_schema_extra={"example": 250.10})
    AverageBasketSize: float = Field(..., description="Average items per transaction", json_schema_extra={"example": 15.2})
    UniqueProducts: int = Field(..., description="Number of unique product codes purchased", json_schema_extra={"example": 45})
    CustomerLifetime: float = Field(..., description="Days between first and last purchase", json_schema_extra={"example": 365.0})
    PurchaseInterval: float = Field(..., description="Average days between purchases", json_schema_extra={"example": 73.0})
    TransactionsPerMonth: float = Field(..., description="Purchase frequency normalized per month", json_schema_extra={"example": 0.41})
    ProductDiversity: float = Field(..., description="Ratio of unique products to total quantity", json_schema_extra={"example": 0.05})

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Active, 1 = Churned", json_schema_extra={"example": 0})
    probability: float = Field(..., description="Probability of churn (0.0 to 1.0)", json_schema_extra={"example": 0.125})
    confidence_score: float = Field(..., description="Confidence score of prediction (0.0 to 1.0)", json_schema_extra={"example": 0.875})
