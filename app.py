import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ============================================
# LOAD SAVED MODEL ARTIFACTS
# ============================================

# Find the project directory
BASE_DIR = Path(__file__).resolve().parent

# Locate the saved model file
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "customer_recommendation_artifacts.joblib"
)

# Load artifacts only once and cache them
@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH)

artifacts = load_artifacts()

# Retrieve saved objects
rfm = artifacts["rfm"]
df_clean = artifacts["df_clean"]
product_model = artifacts["product_model"]
product_user_matrix = artifacts["product_user_matrix"]
product_ids = artifacts["product_ids"]
product_lookup = artifacts["product_lookup"]

# ============================================
# DASHBOARD HEADER
# ============================================

st.title("🛍️ AI-Powered Customer Segmentation")
st.subheader("Customer Intelligence & Product Recommendation System")

st.write(
    "Analyse customer purchasing behaviour, "
    "identify customer segments, and generate "
    "personalised product recommendations."
)

st.success("Models and datasets loaded successfully!")

# Display basic project information
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{len(rfm):,}")
col2.metric("Total Products", f"{len(product_ids):,}")
col3.metric("Total Transactions", f"{len(df_clean):,}")


# ============================================
# CUSTOMER SEARCH
# ============================================

st.divider()
st.header("Customer Analysis")

st.write(
    "Enter a Customer ID to view their purchasing "
    "behaviour, customer segment, and product recommendations."
)

customer_id_input = st.number_input(
    "Enter Customer ID",
    min_value=0,
    value=12347,
    step=1
)

customer_id = float(customer_id_input)

if st.button("Analyse Customer", type="primary"):

    if customer_id not in rfm.index:

        st.error("Customer ID not found. Please enter a valid ID.")

    else:

        # Retrieve customer information
        customer = rfm.loc[customer_id]

        recency = int(customer["Recency"])
        frequency = int(customer["Frequency"])
        monetary = float(customer["Monetary"])
        segment = customer["Customer_Segment"]

        # Retrieve the customer's purchase history
        customer_purchases = df_clean[
            df_clean["CustomerID"] == customer_id
        ].copy()

        purchased_products = set(
            customer_purchases["StockCode"]
        )

        # ============================================
        # CUSTOMER PROFILE
        # ============================================

        st.subheader("Customer Profile")

        st.write("Customer Segment:", segment)

        col1, col2, col3 = st.columns(3)

        col1.metric("Recency", f"{recency} days")
        col2.metric("Frequency", f"{frequency} invoices")
        col3.metric("Monetary", f"£{monetary:,.2f}")

        st.metric(
            "Previously Purchased Products",
            len(purchased_products)
        )

        # ============================================
        # BUSINESS RECOMMENDATIONS
        # ============================================

        business_actions = {
            "Champions": "Offer loyalty rewards and exclusive products.",
            "At-Risk Customers": "Launch re-engagement campaigns and special offers.",
            "Recent Customers": "Encourage repeat purchases through product bundles.",
            "Regular Customers": "Promote complementary products and loyalty benefits."
        }

        st.subheader("Suggested Business Action")

        st.info(
            business_actions.get(
                segment,
                "Review customer purchasing behaviour."
            )
        )

        # ============================================
        # PERSONALised PRODUCT RECOMMENDATIONS
        # ============================================

        st.subheader("Personalised Product Recommendations")

        product_position_lookup = {
            code: position
            for position, code in enumerate(product_ids)
        }

        seed_positions = [
            product_position_lookup[code]
            for code in purchased_products
            if code in product_position_lookup
        ]

        recommendation_scores = {}

        if seed_positions:

            distances, indices = product_model.kneighbors(
                product_user_matrix[seed_positions],
                n_neighbors=min(11, len(product_ids))
            )

            for row_number, seed_position in enumerate(seed_positions):

                for distance, index in zip(
                    distances[row_number],
                    indices[row_number]
                ):

                    if index == seed_position:
                        continue

                    recommended_code = product_ids[index]

                    if recommended_code in purchased_products:
                        continue

                    similarity = max(0, 1 - float(distance))

                    recommendation_scores[recommended_code] = (
                        recommendation_scores.get(recommended_code, 0)
                        + similarity
                    )

        ranked_products = sorted(
            recommendation_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        recommendations = []

        for product_code, score in ranked_products[:5]:

            recommendations.append({
                "Product Code": product_code,
                "Description": product_lookup.get(
                    product_code,
                    "Description unavailable"
                ),
                "Recommendation Score": round(score, 4)
            })

        if recommendations:

            recommendations_df = pd.DataFrame(recommendations)

            st.dataframe(
                recommendations_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No personalised recommendations are available "
                "for this customer."
            )

        # ============================================
        # RECENT PURCHASE HISTORY
        # ============================================

        st.subheader("Recent Purchase History")

        recent_purchases = customer_purchases.sort_values(
            "InvoiceDate",
            ascending=False
        ).head(10)

        st.dataframe(
            recent_purchases[
                [
                    "InvoiceNo",
                    "InvoiceDate",
                    "Description",
                    "Quantity",
                    "UnitPrice",
                    "TotalPrice"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================
# OVERALL CUSTOMER SEGMENT ANALYTICS
# ============================================

st.divider()
st.header("Customer Segment Analytics")

st.write(
    "Explore customer distribution, revenue contribution, "
    "and spending behaviour across different segments."
)

# Create a summary of all customer segments
segment_summary = (
    rfm.groupby("Customer_Segment")
    .agg(
        Customer_Count=("Customer_Segment", "count"),
        Total_Revenue=("Monetary", "sum"),
        Average_Spending=("Monetary", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Recency=("Recency", "mean")
    )
    .reset_index()
)

# ============================================
# CUSTOMER DISTRIBUTION
# ============================================

st.subheader("Customer Distribution by Segment")

fig_customers = px.pie(
    segment_summary,
    names="Customer_Segment",
    values="Customer_Count",
    hole=0.45,
    title="Customer Distribution"
)

fig_customers.update_traces(
    textposition="inside",
    textinfo="percent+label"
)

st.plotly_chart(
    fig_customers,
    use_container_width=True
)

# ============================================
# REVENUE CONTRIBUTION
# ============================================

st.subheader("Revenue Contribution by Segment")

fig_revenue = px.bar(
    segment_summary,
    x="Customer_Segment",
    y="Total_Revenue",
    color="Customer_Segment",
    title="Total Revenue Generated by Each Segment",
    labels={
        "Customer_Segment": "Customer Segment",
        "Total_Revenue": "Revenue (£)"
    },
    text_auto=".2s"
)

st.plotly_chart(
    fig_revenue,
    use_container_width=True
)

# ============================================
# AVERAGE CUSTOMER SPENDING
# ============================================

st.subheader("Average Customer Spending")

fig_spending = px.bar(
    segment_summary,
    x="Customer_Segment",
    y="Average_Spending",
    color="Customer_Segment",
    title="Average Spending per Customer",
    labels={
        "Customer_Segment": "Customer Segment",
        "Average_Spending": "Average Spending (£)"
    },
    text_auto=".2s"
)

st.plotly_chart(
    fig_spending,
    use_container_width=True
)

# ============================================
# SEGMENT SUMMARY TABLE
# ============================================

st.subheader("Segment Performance Summary")

display_summary = segment_summary.copy()

display_summary["Total_Revenue"] = (
    display_summary["Total_Revenue"].round(2)
)

display_summary["Average_Spending"] = (
    display_summary["Average_Spending"].round(2)
)

display_summary["Average_Frequency"] = (
    display_summary["Average_Frequency"].round(2)
)

display_summary["Average_Recency"] = (
    display_summary["Average_Recency"].round(2)
)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True
)



# ============================================
# MODEL PERFORMANCE EVALUATION
# ============================================

st.divider()
st.header("Model Performance Evaluation")

st.write(
    "Evaluation results from the offline testing "
    "of our customer segmentation and recommendation models."
)

# ============================================
# RECOMMENDATION MODEL COMPARISON
# ============================================

st.subheader("Recommendation System Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Collaborative Filtering",
        "Popularity Baseline"
    ],
    "Precision@5": [0.0667, 0.0320],
    "Recall@5": [0.0295, 0.0107],
    "Catalogue Coverage (%)": [11.81, 0.80]
})

col1, col2 = st.columns(2)

col1.metric(
    "Collaborative Filtering Precision@5",
    "6.67%"
)

col2.metric(
    "Popularity Baseline Precision@5",
    "3.20%"
)

fig_comparison = px.bar(
    comparison,
    x="Model",
    y="Precision@5",
    color="Model",
    title="Recommendation Precision@5 Comparison",
    text_auto=".2%"
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.info(
    "Collaborative filtering outperformed the popularity baseline "
    "on all three measured metrics. However, absolute recommendation "
    "accuracy remains low and further improvement is possible."
)

# ============================================
# CLUSTERING MODEL EVALUATION
# ============================================

st.subheader("K-Means Clustering Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("Number of Clusters", "4")
col2.metric("Silhouette Score", "0.3375")
col3.metric("Davies-Bouldin Index", "1.0086")

st.metric(
    "Calinski-Harabasz Score",
    "3,328.34"
)

st.caption(
    "Four clusters were selected for business interpretability. "
    "The two-cluster solution previously achieved a higher "
    "silhouette score."
)

# ============================================
# EVALUATION METHODOLOGY
# ============================================

with st.expander("View Evaluation Methodology"):

    st.write(
        "Recommendation evaluation used a chronological "
        "training-testing split dated 9 November 2011."
    )

    st.write(
        "The evaluation tested 300 sampled eligible customers "
        "and compared collaborative filtering against a "
        "popularity-based baseline."
    )

    st.write(
        "Clustering evaluation used Silhouette Score, "
        "Davies-Bouldin Index, and Calinski-Harabasz Score."
    )

    st.warning(
        "These are offline historical evaluation results. "
        "They do not guarantee future customer purchases."
    )

