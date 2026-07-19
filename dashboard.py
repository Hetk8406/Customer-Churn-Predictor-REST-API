import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Minimal, Business Analytics style)
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
    }
    .metric-val {
        font-size: 24px;
        font-weight: bold;
        color: #1A365D;
    }
    .metric-lbl {
        font-size: 14px;
        color: #4A5568;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 💾 Cacheable Data & Model Loading
# ----------------------------------------------------
@st.cache_data
def load_customer_data():
    if os.path.exists("customer_churn_dataset.csv"):
        return pd.read_csv("customer_churn_dataset.csv")
    return None

@st.cache_data
def load_transaction_data():
    if os.path.exists("online_retail_II.csv"):
        # Load sample or full to save memory in dashboard
        df = pd.read_csv("online_retail_II.csv")
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalAmount'] = df['Quantity'] * df['Price']
        return df
    return None

def load_prediction_model():
    model_path = "api/model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load datasets
df_customer = load_customer_data()
df_transactions = load_transaction_data()
model = load_prediction_model()

# ----------------------------------------------------
# 📌 Sidebar Navigation & Project Info
# ----------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Overview",
        "EDA Dashboard",
        "RFM Analysis",
        "Model Performance",
        "Feature Importance",
        "Customer Prediction",
        "Business Insights",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Information")
st.sidebar.markdown("""
- **Problem**: Customer Churn Prediction
- **Framework**: RFM Analysis + XGBoost
- **API**: FastAPI Deployment
- **Dataset**: Online Retail II (UCI)
""")

# Helper to render clean metric card
def render_metric(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">{label}</div>
        <div class="metric-val">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 1️⃣ Page: Overview
# ----------------------------------------------------
if page == "Overview":
    st.title("Customer Churn Analytics Overview")
    st.markdown("Executive summary of our online retail customer metrics, base counts, and churn rates.")
    
    if df_customer is not None:
        # Calculate statistics
        total_customers = len(df_customer)
        churned_customers = df_customer['Churn'].sum()
        active_customers = total_customers - churned_customers
        churn_rate = (churned_customers / total_customers) * 100
        avg_revenue = df_customer['Monetary'].mean()
        avg_frequency = df_customer['Frequency'].mean()
        avg_recency = df_customer['Recency'].mean()

        # Render KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric("Total Customers", f"{total_customers:,}")
        with col2:
            render_metric("Churn Rate", f"{churn_rate:.2f}%")
        with col3:
            render_metric("Active / Churned", f"{active_customers:,} / {churned_customers:,}")
        with col4:
            render_metric("Avg Customer Value", f"${avg_revenue:,.2f}")

        st.markdown("---")
        
        # Plotly Churn Balance chart
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Customer Class Distribution")
            fig = px.pie(
                df_customer,
                names=df_customer['Churn'].map({0: 'Active (0)', 1: 'Churned (1)'}),
                color_discrete_sequence=['#3182CE', '#E53E3E'],
                hole=0.4
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Inactivity distribution (Recency)")
            fig = px.histogram(
                df_customer,
                x='Recency',
                color=df_customer['Churn'].map({0: 'Active', 1: 'Churned'}),
                color_discrete_map={'Active': '#3182CE', 'Churned': '#E53E3E'},
                nbins=40,
                barmode='overlay'
            )
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# 2️⃣ Page: EDA Dashboard
# ----------------------------------------------------
elif page == "EDA Dashboard":
    st.title("E-Commerce Transaction EDA Dashboard")
    st.markdown("Historical transaction-level statistics over the 2-year duration.")

    if df_transactions is not None:
        # Time-series groups
        df_transactions['InvoiceMonth'] = df_transactions['InvoiceDate'].dt.to_period('M').astype(str)
        monthly_sales = df_transactions.groupby('InvoiceMonth').agg(
            Revenue=('TotalAmount', 'sum'),
            Invoices=('Invoice', 'nunique')
        ).reset_index()

        tab1, tab2, tab3 = st.tabs(["Monthly Trends", "Geographic Base", "Popular Items"])
        
        with tab1:
            st.subheader("Monthly Revenue & Invoice Counts")
            fig1 = px.line(monthly_sales, x='InvoiceMonth', y='Revenue', title="Monthly Sales Revenue ($)", markers=True)
            fig1.update_traces(line_color="#2B6CB0")
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = px.line(monthly_sales, x='InvoiceMonth', y='Invoices', title="Monthly Transaction Count", markers=True)
            fig2.update_traces(line_color="#319795")
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.subheader("Top Countries by Transactions")
            col_l, col_r = st.columns(2)
            with col_l:
                top_countries = df_transactions['Country'].value_counts().head(10).reset_index()
                top_countries.columns = ['Country', 'Transactions']
                fig3 = px.bar(top_countries, y='Country', x='Transactions', orientation='h', title="Including UK", color_discrete_sequence=['#4299E1'])
                fig3.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig3, use_container_width=True)
            with col_r:
                top_countries_no_uk = df_transactions[df_transactions['Country'] != 'United Kingdom']['Country'].value_counts().head(10).reset_index()
                top_countries_no_uk.columns = ['Country', 'Transactions']
                fig4 = px.bar(top_countries_no_uk, y='Country', x='Transactions', orientation='h', title="Excluding UK", color_discrete_sequence=['#DD6B20'])
                fig4.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig4, use_container_width=True)

        with tab3:
            st.subheader("Top 15 Selling Products by Quantity")
            top_products = df_transactions.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(15).reset_index()
            fig5 = px.bar(top_products, y='Description', x='Quantity', orientation='h', color_discrete_sequence=['#805AD5'])
            fig5.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig5, use_container_width=True)

# ----------------------------------------------------
# 3️⃣ Page: RFM Analysis
# ----------------------------------------------------
elif page == "RFM Analysis":
    st.title("Customer RFM Distribution")
    st.markdown("Inspect distributions, correlations, and segments of customer-level features.")

    if df_customer is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Frequency distribution (Number of Invoices)")
            fig_freq = px.histogram(df_customer[df_customer['Frequency'] < 30], x='Frequency', color_discrete_sequence=['#319795'])
            st.plotly_chart(fig_freq, use_container_width=True)
        with col2:
            st.subheader("Monetary distribution (Total Spend)")
            fig_mon = px.histogram(df_customer[df_customer['Monetary'] < 5000], x='Monetary', color_discrete_sequence=['#DD6B20'])
            st.plotly_chart(fig_mon, use_container_width=True)

        st.markdown("---")
        
        # Heatmap of correlations
        st.subheader("Correlation Heatmap of Customer Metrics")
        corr_matrix = df_customer.drop(columns=['Customer ID']).corr()
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="Blues",
            aspect="auto"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ----------------------------------------------------
# 4️⃣ Page: Model Performance
# ----------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance Metrics")
    st.markdown("Validation performance results for the baseline models and XGBoost.")

    # Model scores (Pre-calculated from notebook runs)
    scores = {
        "Model": ["Random Forest", "XGBoost", "Decision Tree", "Logistic Regression"],
        "Accuracy": [1.000, 1.000, 1.000, 0.992],
        "Precision": [1.000, 1.000, 1.000, 1.000],
        "Recall": [1.000, 1.000, 1.000, 0.985],
        "F1-Score": [1.000, 1.000, 1.000, 0.992],
        "ROC-AUC": [1.000, 1.000, 1.000, 0.999]
    }
    df_scores = pd.DataFrame(scores)
    st.dataframe(df_scores, use_container_width=True)

    st.markdown("---")
    
    st.subheader("Validation Heatmap (Confusion Matrix Heuristics)")
    st.info("The tree-based models achieve perfect score matrices due to the deterministic target threshold split on the Recency metric.")

# ----------------------------------------------------
# 5️⃣ Page: Feature Importance
# ----------------------------------------------------
elif page == "Feature Importance":
    st.title("Feature Importance Analysis")
    st.markdown("Feature importance distributions according to XGBoost split-gains and SHAP game values.")

    # Static feature importances for Plotly from model outputs
    features = [
        "Recency", "Customer_Lifetime", "Transactions_Per_Month", 
        "Purchase_Interval", "Frequency", "Product_Diversity", 
        "Average_Basket_Size", "Unique_Products", "Monetary", "Average_Order_Value"
    ]
    xgb_gain = [0.85, 0.08, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.001, 0.000]
    
    df_fi = pd.DataFrame({"Feature": features, "XGBoost Gain": xgb_gain})
    
    fig = px.bar(df_fi, x='XGBoost Gain', y='Feature', orientation='h', color_discrete_sequence=['#3182CE'], title="XGBoost Feature Importance (Gain)")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# 6️⃣ Page: Customer Prediction
# ----------------------------------------------------
elif page == "Customer Prediction":
    st.title("Run Customer Churn Prediction")
    st.markdown("Input customer-level parameters to evaluate the likelihood of churn.")

    if model is None:
        st.error("Production model 'model.joblib' is not loaded or missing from the api directory.")
    else:
        # Form for input
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                recency = st.number_input("Recency (Days since last purchase)", min_value=0, max_value=730, value=45)
                frequency = st.number_input("Frequency (Invoice Count)", min_value=1, max_value=500, value=5)
                monetary = st.number_input("Monetary (Total Spend, $)", min_value=0.0, max_value=1000000.0, value=1250.0)
                aov = st.number_input("Average Order Value (Monetary / Frequency, $)", min_value=0.0, value=250.0)
                basket_size = st.number_input("Average Basket Size (Total Quantity / Frequency)", min_value=0.0, value=15.0)
            with col2:
                unique_products = st.number_input("Unique Products (Count of unique StockCodes)", min_value=1, value=45)
                lifetime = st.number_input("Customer Lifetime (Days between first/last purchase)", min_value=0, max_value=730, value=365)
                interval = st.number_input("Purchase Interval (Lifetime / Frequency, Days)", min_value=0.0, value=73.0)
                tx_per_month = st.number_input("Transactions Per Month", min_value=0.0, value=0.41)
                diversity = st.number_input("Product Diversity (Unique StockCodes / Total Quantity)", min_value=0.0, value=0.05)

            submit = st.form_submit_button("Predict Churn")

        if submit:
            # Map input parameters to DataFrame matching training layout
            input_df = pd.DataFrame([{
                "Recency": float(recency),
                "Frequency": int(frequency),
                "Monetary": float(monetary),
                "Average_Order_Value": float(aov),
                "Average_Basket_Size": float(basket_size),
                "Unique_Products": int(unique_products),
                "Customer_Lifetime": float(lifetime),
                "Purchase_Interval": float(interval),
                "Transactions_Per_Month": float(tx_per_month),
                "Product_Diversity": float(diversity)
            }])

            # Perform prediction
            pred = int(model.predict(input_df)[0])
            prob = float(model.predict_proba(input_df)[0][1])
            confidence = prob if pred == 1 else 1.0 - prob

            st.markdown("---")
            st.subheader("Prediction Results")

            c1, c2, c3 = st.columns(3)
            with c1:
                status_label = "CHURNED (1)" if pred == 1 else "ACTIVE (0)"
                st.metric("Prediction", status_label)
            with c2:
                st.metric("Churn Probability", f"{prob:.2%}")
            with c3:
                st.metric("Confidence Score", f"{confidence:.2%}")

            # Recommendations based on risk
            st.markdown("### Actionable Business Recommendation")
            if prob < 0.3:
                st.success("🟢 **LOW RISK**: The customer is highly active and engaged. Recommendation: Include in standard product updates and reward with baseline loyalty points.")
            elif 0.3 <= prob < 0.7:
                st.warning("🟡 **MEDIUM RISK**: Customer shows signs of drifting. Recommendation: Send targeted re-engagement emails, offer feedback surveys, or trigger mid-tier discount incentives.")
            else:
                st.error("🔴 **HIGH RISK**: The customer has likely churned or is about to. Recommendation: Trigger premium win-back promotions, personal client outreach, or value-driven re-engagement campaigns.")

# ----------------------------------------------------
# 7️⃣ Page: Business Insights
# ----------------------------------------------------
elif page == "Business Insights":
    st.title("Business Insights & Strategy Recommendations")
    
    st.subheader("Top Churn Drivers (According to SHAP Audit)")
    st.markdown("""
    1. **Inactivity Duration (Recency)**: The number of days since the last purchase is the single most critical predictor. Active management is key before a customer passes 60 days of silence.
    2. **Short Lifetime**: Customers who buy once or twice and show lifetimes under 5 days are highly likely to drop off permanently, indicating a poor onboarding experience.
    3. **Purchase Frequency**: A low transaction frequency (1–3 invoices over 2 years) reflects lack of customer purchase habituation.
    """)

    st.subheader("Strategy & Retention Recommendations")
    st.markdown("""
    - **Onboarding Interventions**: For new customers (low lifetime), trigger automated email welcome flows offering second-purchase discounts.
    - **Proactive Reactivation**: Implement warning alerts for customers whose recency rises past 45 days. Do not wait for the 90-day churn threshold to act.
    - **VIP Retention**: Heavy spenders (high Monetary) who show an increasing purchase interval should receive dedicated customer support calls or premium custom perks.
    """)

# ----------------------------------------------------
# 8️⃣ Page: About Project
# ----------------------------------------------------
elif page == "About Project":
    st.title("About Customer Churn Predictor Project")
    st.markdown("""
    This project is an end-to-end data science portfolio project focusing on customer lifetime analytics and deployment.
    
    ### 💻 Technology Stack:
    - **Back-end & Modeling**: Python, Scikit-learn, XGBoost, SHAP, Joblib
    - **REST API**: FastAPI, Uvicorn, Pydantic
    - **Dashboard**: Streamlit, Plotly
    
    ### 👨‍💻 Developer & Authorship:
    - **Author**: Het Kikani
    - **GitHub**: [Customer Churn Predictor Repository](https://github.com/Hetk8406/Customer-Churn-Predictor-REST-API)
    - **LinkedIn**: [Het Kikani Profile](https://www.linkedin.com/in/het-kikani-67817236b/)
    """)
