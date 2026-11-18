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

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "customer_recommendation_artifacts.joblib"
)

# Required artifacts
REQUIRED_ARTIFACTS = [
    "rfm",
    "df_clean",
    "product_model",
    "product_user_matrix",
    "product_ids",
    "product_lookup"
]

# Load artifacts only once
@st.cache_resource
def load_artifacts():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact file not found: {MODEL_PATH}"
        )

    loaded = joblib.load(MODEL_PATH)

    missing = [
        key for key in REQUIRED_ARTIFACTS
        if key not in loaded
    ]

    if missing:
        raise ValueError(
            f"Missing required artifacts: {missing}"
        )

    return loaded


try:
    artifacts = load_artifacts()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

except Exception as error:
    st.error("Unable to load model artifacts.")
    st.exception(error)
    st.stop()


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

st.success("Model artifacts loaded successfully!")

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

        st.info(f"Customer Segment: {segment}")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Recency",
            f"{recency} days"
        )

        col2.metric(
            "Frequency",
            f"{frequency} invoices"
        )

        col3.metric(
            "Monetary",
            f"£{monetary:,.2f}"
        )

        col4.metric(
            "Unique Products Purchased",
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
        # PERSONALISED PRODUCT RECOMMENDATIONS
        # ============================================

        st.subheader("Personalised Product Recommendations")

        st.write(
            "Products suggested based on similarities "
            "between items in the customer's purchase history."
        )

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
                "Product Description": product_lookup.get(
                    product_code,
                    "Description unavailable"
                ),
                "Similarity Score": round(score, 4)
            })

        if recommendations:

            recommendations_df = pd.DataFrame(recommendations)

            st.success(
                f"{len(recommendations)} personalised recommendations generated."
            )

            st.dataframe(
                recommendations_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Product Code": st.column_config.TextColumn(
                        "Product Code"
                    ),
                    "Product Description": st.column_config.TextColumn(
                        "Product Description",
                        width="large"
                    ),
                    "Similarity Score": st.column_config.NumberColumn(
                        "Similarity Score",
                        format="%.4f"
                    )
                }
            )

            st.caption(
                "Higher scores indicate stronger accumulated similarity "
                "with the customer's previously purchased products. "
                "These scores are not purchase probabilities."
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

        total_invoices = customer_purchases["InvoiceNo"].nunique()
        total_purchase_value = customer_purchases["TotalPrice"].sum()

        history_col1, history_col2 = st.columns(2)

        history_col1.metric(
            "Total Invoices",
            f"{total_invoices:,}"
        )

        history_col2.metric(
            "Total Purchase Value",
            f"£{total_purchase_value:,.2f}"
        )

        recent_purchases = (
            customer_purchases
            .sort_values("InvoiceDate", ascending=False)
            .head(10)
            .copy()
        )

        recent_purchases["InvoiceDate"] = (
            recent_purchases["InvoiceDate"].dt.strftime("%d %b %Y")
        )

        st.caption("Showing the 10 most recent product purchase records.")

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
            hide_index=True,
            column_config={
                "InvoiceNo": st.column_config.TextColumn(
                    "Invoice Number"
                ),
                "InvoiceDate": st.column_config.TextColumn(
                    "Invoice Date"
                ),
                "Description": st.column_config.TextColumn(
                    "Product Description",
                    width="large"
                ),
                "Quantity": st.column_config.NumberColumn(
                    "Quantity",
                    format="%d"
                ),
                "UnitPrice": st.column_config.NumberColumn(
                    "Unit Price",
                    format="£%.2f"
                ),
                "TotalPrice": st.column_config.NumberColumn(
                    "Total Price",
                    format="£%.2f"
                )
            }
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
# AVERAGE PURCHASE FREQUENCY
# ============================================

st.subheader("Average Purchase Frequency by Segment")

fig_frequency = px.bar(
    segment_summary,
    x="Customer_Segment",
    y="Average_Frequency",
    color="Customer_Segment",
    title="Average Number of Invoices per Customer",
    labels={
        "Customer_Segment": "Customer Segment",
        "Average_Frequency": "Average Invoices"
    },
    text_auto=".2f"
)

st.plotly_chart(
    fig_frequency,
    use_container_width=True
)

# ============================================
# AVERAGE RECENCY
# ============================================

st.subheader("Average Recency by Segment")

fig_recency = px.bar(
    segment_summary,
    x="Customer_Segment",
    y="Average_Recency",
    color="Customer_Segment",
    title="Average Days Since Last Purchase",
    labels={
        "Customer_Segment": "Customer Segment",
        "Average_Recency": "Average Recency (Days)"
    },
    text_auto=".2f"
)

st.plotly_chart(
    fig_recency,
    use_container_width=True
)

st.caption(
    "Lower recency values indicate customers who purchased more recently. "
    "Higher values may indicate customers who need re-engagement."
)

# ============================================
# SEGMENT SUMMARY TABLE
# ============================================

st.subheader("Segment Performance Summary")

display_summary = segment_summary.rename(
    columns={
        "Customer_Segment": "Customer Segment",
        "Customer_Count": "Number of Customers",
        "Total_Revenue": "Total Revenue (£)",
        "Average_Spending": "Average Spending (£)",
        "Average_Frequency": "Average Invoices",
        "Average_Recency": "Average Recency (Days)"
    }
).copy()

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Customer Segment": st.column_config.TextColumn(
            "Customer Segment"
        ),
        "Number of Customers": st.column_config.NumberColumn(
            "Number of Customers",
            format="%d"
        ),
        "Total Revenue (£)": st.column_config.NumberColumn(
            "Total Revenue (£)",
            format="£%.2f"
        ),
        "Average Spending (£)": st.column_config.NumberColumn(
            "Average Spending (£)",
            format="£%.2f"
        ),
        "Average Invoices": st.column_config.NumberColumn(
            "Average Invoices",
            format="%.2f"
        ),
        "Average Recency (Days)": st.column_config.NumberColumn(
            "Average Recency (Days)",
            format="%.2f"
        )
    }
)



# ============================================
# MODEL PERFORMANCE EVALUATION
# ============================================

st.divider()
st.header("Model Performance Evaluation")

st.write(
    "This section presents the offline evaluation of the "
    "recommendation system and K-Means customer segmentation model."
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
    "Precision@5 (%)": [6.67, 3.20],
    "Recall@5 (%)": [2.95, 1.07],
    "Catalogue Coverage (%)": [11.81, 0.80]
})

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "CF Precision@5",
    "6.67%",
    "vs 3.20% baseline"
)

metric_col2.metric(
    "CF Recall@5",
    "2.95%",
    "vs 1.07% baseline"
)

metric_col3.metric(
    "CF Catalogue Coverage",
    "11.81%",
    "vs 0.80% baseline"
)

comparison_long = comparison.melt(
    id_vars="Model",
    var_name="Evaluation Metric",
    value_name="Score (%)"
)

fig_comparison = px.bar(
    comparison_long,
    x="Evaluation Metric",
    y="Score (%)",
    color="Model",
    barmode="group",
    title="Collaborative Filtering vs Popularity Baseline",
    text_auto=".2f",
    labels={
        "Evaluation Metric": "Evaluation Metric",
        "Score (%)": "Score (%)",
        "Model": "Recommendation Model"
    }
)

fig_comparison.update_layout(
    yaxis_title="Score (%)",
    xaxis_title="",
    legend_title="Model"
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Model": st.column_config.TextColumn("Model"),
        "Precision@5 (%)": st.column_config.NumberColumn(
            "Precision@5",
            format="%.2f%%"
        ),
        "Recall@5 (%)": st.column_config.NumberColumn(
            "Recall@5",
            format="%.2f%%"
        ),
        "Catalogue Coverage (%)": st.column_config.NumberColumn(
            "Catalogue Coverage",
            format="%.2f%%"
        )
    }
)

st.info(
    "Collaborative filtering performed better than the popularity "
    "baseline across all three measured metrics. However, the absolute "
    "precision and recall values are still low, so recommendations "
    "should be treated as suggestions rather than guaranteed purchases."
)

# ============================================
# CLUSTERING MODEL EVALUATION
# ============================================

st.subheader("K-Means Clustering Evaluation")

cluster_col1, cluster_col2, cluster_col3 = st.columns(3)

cluster_col1.metric("Number of Clusters", "4")
cluster_col2.metric("Silhouette Score", "0.3375")
cluster_col3.metric("Davies-Bouldin Index", "1.0086")

st.metric(
    "Calinski-Harabasz Score",
    "3,328.34"
)

st.caption(
    "The four-cluster solution was selected for business interpretability. "
    "A two-cluster solution had a higher silhouette score (approximately "
    "0.433), so four clusters should not be described as the mathematically "
    "optimal solution."
)

# ============================================
# EVALUATION METHODOLOGY
# ============================================

with st.expander("View Evaluation Methodology"):

    st.markdown("**Recommendation system evaluation**")

    st.write(
        "A chronological split was used to reduce data leakage. "
        "The training data covered 1 December 2010 to 8 November 2011, "
        "and the test data covered 9 November 2011 to 9 December 2011."
    )

    st.write(
        "The recommender was trained using training-period interactions. "
        "Evaluation used a sample of 300 eligible customers who had "
        "purchases in both the training and test periods."
    )

    st.write(
        "Precision@5 measures the proportion of the five recommended "
        "items that appeared in the customer's test-period purchases. "
        "Recall@5 measures the proportion of relevant test-period items "
        "covered by the five recommendations. Catalogue coverage measures "
        "the share of the eligible product catalogue recommended."
    )

    st.markdown("**Clustering evaluation**")

    st.write(
        "The K-Means model was evaluated using Silhouette Score, "
        "Davies-Bouldin Index, and Calinski-Harabasz Score. Four clusters "
        "were retained because their customer profiles were useful for "
        "business interpretation, despite the higher silhouette score "
        "observed with two clusters."
    )

    st.warning(
        "Limitations: recommendation metrics were calculated on a sample "
        "of 300 eligible customers and a roughly 30-day test period. "
        "The final dataset ends on 9 December 2011, so these historical "
        "offline results do not establish performance on current customers "
        "or guarantee future purchases."
    )
