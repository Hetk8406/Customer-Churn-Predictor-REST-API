# Customer Churn Predictor: 20 Interview Questions & Detailed Answers

This document contains 20 highly detailed, natural, and professional interview questions with comprehensive answers covering all aspects of the Customer Churn prediction project.

---

### Topic 1: Customer Churn & Business Decisions

#### Q1: What is the difference between churn in a subscription-based business model vs. a transaction-based business model, and how does it affect label creation?
**Answer:**  
In subscription models (like Netflix or SaaS), churn is explicit: a customer clicks "cancel subscription," which records an exact timestamp. In non-subscription transaction models (like retail, e-commerce, or hospitality), churn is implicit. Customers don't cancel; they simply stop buying. To define churn, we must establish a temporal threshold of inactivity (e.g., 90 days). If a customer hasn't purchased within that window, we label them as churned. The choice of window is crucial: too short and you flag active customers as churned (high false positives); too long and you react too late to retain them.

#### Q2: Why is a 90-day window appropriate for this retail dataset, and how would you validate this choice in a real-world business context?
**Answer:**  
For general retail, a purchase cycle typically repeats every 30 to 60 days. A 90-day window represents three months of complete inactivity, which is a reasonable heuristic suggesting the customer has moved to a competitor. To validate this choice, we would perform a **Recency distribution analysis** or calculate the **95th percentile of the customer-level purchase interval**. If 95% of active, repeat customers repurchase within 75 days, then a 90-day window is statistically sound because a gap longer than 90 days represents a statistically significant deviation from normal purchasing habits.

#### Q3: Explain the concept of target leakage in customer churn prediction. Did we have leakage in this project?
**Answer:**  
Target leakage occurs when information from the future (or information directly related to the target variable) is included in the feature set during training, leading to unrealistically high model performance. In this project, we defined `Churn` mathematically as `Recency > 90`. Because we kept `Recency` in the features, the tree models easily found a split on `Recency` at 90 days and achieved perfect 1.0 accuracy. This is a classic case of target leakage. In production, `Recency` must be excluded from the model features, forcing it to predict churn based on underlying behaviors (like frequency, AOV, or basket size) rather than the recency metric itself.

#### Q4: How can the marketing team use the model's outputs (probability scores and explanations) to optimize their retention budget?
**Answer:**  
Instead of sending a blanket discount to all customers, which is expensive and erodes margins, the marketing team can prioritize their budget:
1. **Targeting by Probability**: Target customers with mid-to-high probability (e.g., 0.6 to 0.8) who are sliding towards churn but still reachable.
2. **Exclusion of 'Dead' Customers**: Do not spend budget on customers with a probability of 0.99 who have been inactive for over a year (unrecoverable).
3. **Tailored Campaigns via SHAP**: If SHAP shows a customer is churning due to a drop in spending size, send a high-value product discount. If they are churning because of a long purchase interval, send a re-engagement coupon.

---

### Topic 2: RFM & Feature Engineering

#### Q5: Walk me through the mathematical formulation of RFM metrics. How do you handle cases where a customer has only one purchase?
**Answer:**  
- **Recency**: Days between the customer's last purchase date and a reference date (the day after the latest transaction in the dataset). Formula: $R_i = D_{ref} - \max(D_{transactions, i})$.
- **Frequency**: The count of unique invoice identifiers. Formula: $F_i = \text{count}(\text{unique Invoices}_i)$.
- **Monetary**: The sum of transaction values. Formula: $M_i = \sum (Quantity \times Price)_i$.

For single-purchase customers, Recency is calculated normally relative to the reference date. Frequency is 1. Monetary is the value of that single transaction. Their customer lifetime is 1 day, and their purchase interval defaults to their lifetime (or is undefined, which we handle by setting it to a baseline default like 1 to avoid division by zero).

#### Q6: Why did you engineer auxiliary features like Average Order Value, Customer Lifetime, and Product Diversity?
**Answer:**  
Standard RFM only captures high-level aggregates. Additional features add crucial context:
- **Average Order Value (AOV)**: Distinguishes between someone who buys 10 items worth $10 each ($100 total, AOV $10) and someone who buys 1 item worth $100 ($100 total, AOV $100).
- **Customer Lifetime**: Captures relationship depth. A customer active for 700 days is fundamentally different from a customer active for 5 days, even if their frequency is the same.
- **Product Diversity**: Measures buy breadth. A customer buying a single product type in bulk behaves differently from a customer purchasing a wide array of items in small quantities.

#### Q7: What are the main challenges when dealing with skewed distributions in RFM features, and how do you handle them for different models?
**Answer:**  
RFM metrics (especially Frequency and Monetary) are heavily right-skewed with extreme outliers due to commercial wholesale buyers. 
- **For Linear/Distance Models**: Skewed features violate normality assumptions and dominate distance calculations. We apply logarithmic transforms ($\log(x+1)$) or robust scaling.
- **For Tree Models**: Skewness is not an issue because trees split on order-based thresholds. However, extreme values can make finding splits less stable, so capping/winsorizing is sometimes applied.

---

### Topic 3: XGBoost

#### Q8: How does XGBoost differ from Random Forest, and why is it preferred for churn prediction?
**Answer:**  
- **Random Forest** is a bagging ensemble that trains multiple deep decision trees independently in parallel. The final prediction is an average of all trees, which reduces variance.
- **XGBoost** is a gradient boosting framework that trains trees sequentially. Each new tree is trained to correct the residual errors made by the previous trees, optimizing a specific loss function.

XGBoost is often preferred for churn prediction because its sequential boosting focuses on hard-to-classify samples. It supports custom loss functions, handles missing values internally, and includes built-in regularization ($L1$ and $L2$) to prevent overfitting.

#### Q9: What is the role of `scale_pos_weight` in XGBoost, and how does it help with class imbalance?
**Answer:**  
In highly imbalanced datasets (e.g., 5% churn, 95% active), standard classifiers naturally optimize for the majority class. XGBoost's `scale_pos_weight` parameter adjusts the loss function's weight for positive class updates. It is typically set as:
$$\text{scale\_pos\_weight} = \frac{\text{sum(negative cases)}}{\text{sum(positive cases)}}$$
This scales the gradient updates for the minority class, forcing the algorithm to penalize misclassifications of the minority class more heavily, improving minority recall.

#### Q10: Explain the significance of the `gamma` and `max_depth` parameters in controlling XGBoost overfitting.
**Answer:**  
- **`max_depth`**: Restricts the maximum depth of each tree. Deeper trees capture more complex interactions but are prone to memorizing training noise (overfitting). Standard values range from 3 to 10.
- **`gamma`**: Specifies the minimum loss reduction required to make a split on a leaf node. It acts as a regularizer; higher values of gamma make the model more conservative, preventing splits that only offer marginal improvements.

---

### Topic 4: SHAP (Explainability)

#### Q11: Explain the mathematical foundation of SHAP values. Why are they superior to native tree feature importances?
**Answer:**  
SHAP is based on **Shapley values** from cooperative game theory. It treats features as players in a coalition. The SHAP value of a feature is its average marginal contribution to the prediction across all possible feature subsets. 
SHAP is superior because it is **consistent** and **additive**:
- Native tree importances (like Gini gain) can decrease for a feature even if the model becomes more dependent on it when other features are added.
- SHAP values guarantee that the sum of feature attributions equals the difference between the model's prediction and the baseline average prediction, making the explanations mathematically robust.

#### Q12: How do you interpret a SHAP summary plot (bee swarm plot) in the context of customer retention?
**Answer:**  
On a SHAP summary plot:
- The y-axis lists features sorted by overall importance (mean absolute SHAP).
- The x-axis represents the SHAP value (impact on model output). Positive values push the model towards churn (y=1); negative values push towards active (y=0).
- The color represents feature values: red for high, blue for low.

For example, if `Customer_Lifetime` has a long tail on the left with red dots, it means a **high** customer lifetime (red) strongly reduces churn probability (negative SHAP). If `Recency` has a long tail on the right with red dots, it means **high** recency (red) strongly increases churn probability (positive SHAP).

#### Q13: What does a SHAP dependence plot show, and how does it reveal non-linear thresholds?
**Answer:**  
A SHAP dependence plot shows the relationship between a single feature's value (x-axis) and its corresponding SHAP value (y-axis) across all samples. It allows us to view the exact shape of the feature's impact.
In our churn project, plotting `Recency` against its SHAP value shows a flat negative impact up to 90 days, followed by a sharp vertical jump to a positive impact at 90 days. This clearly reveals the non-linear, step-function threshold learned by the model.

---

### Topic 5: Model Evaluation & Validation

#### Q14: Why is the Precision-Recall (PR) Curve preferred over the ROC Curve for evaluating models on imbalanced datasets?
**Answer:**  
- **ROC Curves** plot True Positive Rate (Recall) vs. False Positive Rate (FPR). FPR is calculated as $\frac{FP}{FP + TN}$. In datasets with a massive majority class, the True Negative ($TN$) count is huge, which keeps the FPR artificially low even if the model makes many false positive errors.
- **PR Curves** plot Precision ($\frac{TP}{TP + FP}$) vs. Recall ($\frac{TP}{TP + FN}$). They do not include $TN$ in their calculations, making them highly sensitive to false positives even when the majority class is large. This makes PR curves far more informative when evaluating minority class prediction.

#### Q15: Why is K-Fold Cross-Validation necessary, and how does it help protect against data leakage?
**Answer:**  
K-Fold Cross-Validation splits the dataset into $K$ equal subsets (folds). The model is trained on $K-1$ folds and validated on the remaining fold, repeating the process $K$ times. It ensures the model's performance is stable across different subsets and not just lucky on a single train-test split.
To protect against data leakage during cross-validation, any preprocessing steps (like scaling, imputation, or feature engineering) must be executed *inside* each cross-validation loop using scikit-learn pipelines. Doing preprocessing globally before splitting leaks validation data information into the training phase.

---

### Topic 6: Ethics & Deployment

#### Q16: Describe the architecture of our FastAPI deployment. How does it handle scaling and startup times?
**Answer:**  
Our FastAPI deployment utilizes a lightweight, modular REST architecture:
- **`schemas.py`**: Validates request payloads on arrival using Pydantic, returning a standard 422 error if fields are malformed.
- **`utils.py`**: Formats requests into structured Pandas DataFrames with strict column sorting, and loads the serialized XGBoost model using Joblib.
- **`app.py`**: The application hub. It loads the model **once** during startup to minimize request latency (warm start).

FastAPI runs asynchronously (ASGI) via Uvicorn, enabling it to handle thousands of concurrent requests with low latency, making it highly suitable for high-throughput production environments.

#### Q17: What are the ethical implications of predicting customer churn, and how can biased predictions cause harm?
**Answer:**  
Ethical concerns in churn prediction primarily revolve around bias and discrimination:
1. **Disparate Treatment**: If feature inputs include sensitive demographics (like age, gender, or geographic location), the model might learn to offer discounts only to specific privileged groups, leaving out others.
2. **Predatory Targeting**: Models might target vulnerable segments with high-interest credit or low-value products.
3. **Data Privacy**: Collecting transactional data to predict behavioral drift requires transparent consent and compliance with frameworks like GDPR and CCPA.

#### Q18: What is model drift, and how would you monitor the deployed FastAPI service to detect it?
**Answer:**  
Model drift occurs when the performance of a deployed model degrades over time because the relationship between features and target variables changes (concept drift) or the underlying feature distribution shifts (covariate shift).
To monitor and detect drift:
1. **Performance Logging**: Log API inputs and predictions. Once actual customer labels are known (e.g., 90 days later), calculate accuracy/F1 metrics and compare them to training metrics.
2. **Distribution Tracking**: Use statistical tests (like the Kolmogorov-Smirnov test) to check if current input feature distributions (e.g., Recency or spend patterns) deviate from the training feature distributions.

#### Q19: Why is it critical to validate feature order before passing data to an XGBoost model in production, and how did we implement this in `utils.py`?
**Answer:**  
Machine Learning models (like XGBoost) do not look at feature names; they look at column indexes. If a client sends features in a different JSON order, or if Pydantic compiles a dictionary differently, passing the raw array to the model will lead to mismatched feature values (e.g., feeding Price values into the Quantity column), causing silent, disastrous predictions.
In `api/utils.py`, we implement a strict list `feature_order` containing all feature names in the exact training sequence. The incoming Pydantic request is converted to a dictionary, mapped to training column names, and structured into a Pandas DataFrame using `df[feature_order]` to guarantee column alignment before inference.

#### Q20: If your churn model performs with 100% accuracy in validation but drops to 60% in production, what are the most likely causes?
**Answer:**  
1. **Target Leakage**: The most common culprit. A feature was present in training that is mathematically bound to the label (like `Recency` in our 90-day threshold definition) but is not available or behaves differently at prediction time.
2. **Data Pipeline Differences**: The data preprocessing or cleaning pipeline in production differed from the training pipeline (e.g., handling missing values, currency conversions, or timezone offsets differently).
3. **Selection Bias / Covariate Shift**: The training data was collected from a specific segment (e.g., UK customers during holiday seasons) and does not generalize to the broader global audience in production.
