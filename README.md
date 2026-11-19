# AI-Powered Customer Segmentation & Product Recommendation System

An end-to-end Machine Learning project that analyses customer purchasing behaviour, identifies customer segments using RFM analysis and K-Means clustering, and generates personalised product recommendations using item-based collaborative filtering.

The project also includes an interactive Streamlit dashboard for exploring customer profiles, purchase history, recommendations, and model evaluation results.


## Live Demo

[Launch Demo](https://ai-customer-segmentation-recommendation.streamlit.app/)

## Screenshots

### Dashboard
![Dashboard]("E:\Projects\AI Customer Segmentation\screenshots\dashboard-overview.png")

### Customer Analysis
![Customer Analysis]("E:\Projects\AI Customer Segmentation\screenshots\customer-analysis.png")

### Segment Analytics
![Segment Analytics]("E:\Projects\AI Customer Segmentation\screenshots\segment-analytics.png")

## Project Overview

Understanding customer purchasing behaviour helps businesses improve customer engagement, retention, and product discovery.

This project uses historical retail transaction data to:

- Analyse customer purchasing behaviour.
- Identify customer segments using RFM analysis and K-Means clustering.
- Generate personalised product recommendations using item-based collaborative filtering.
- Compare recommendation performance against a popularity-based baseline.
- Visualise customer insights through an interactive Streamlit dashboard.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Customer Segmentation | K-Means Clustering |
| Recommendation System | Item-Based Collaborative Filtering |
| Similarity Measurement | Cosine Similarity |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| Development Environment | Jupyter Notebook, VS Code |

## Dataset

**Online Retail Dataset – UCI Machine Learning Repository**

Source: https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains transactions from a UK-based online retail store between December 2010 and December 2011.

**Original dataset:**
- 541,909 transaction records
- 8 columns
- 4,372 unique customers

**After data cleaning:**
- 392,692 transaction records
- 4,338 unique customers
- 3,665 unique products

The original dataset is not included in this repository.

## Project Workflow

### 1. Data Cleaning

The following preprocessing operations were performed:

- Duplicate transaction removal.
- Cancellation transaction removal.
- Missing Customer ID removal.
- Invalid quantity and unit price filtering.
- Calculation of total transaction value.

### 2. Exploratory Data Analysis

Explored customer and product purchasing patterns through:

- Customer purchasing behaviour.
- Top-selling products.
- Revenue by country.
- Monthly revenue trends.
- Highest-spending customers.

### 3. RFM Analysis

Customer behaviour was analysed using three features:

- **Recency:** Number of days since the customer's latest purchase.
- **Frequency:** Number of distinct invoices associated with the customer.
- **Monetary:** Total amount spent by the customer.

Log transformation and StandardScaler were applied to prepare the RFM features for clustering.

### 4. Customer Segmentation Using K-Means

K-Means clustering was implemented with four customer segments.

| Customer Segment | Customers | Percentage |
|---|---:|---:|
| Champions | 713 | 16.44% |
| At-Risk Customers | 1,622 | 37.39% |
| Recent Customers | 837 | 19.29% |
| Regular Customers | 1,166 | 26.88% |
| **Total** | **4,338** | **100%** |

These segments help identify different customer behaviours and support targeted business strategies such as loyalty rewards, customer re-engagement, repeat-purchase campaigns, and complementary product promotions.

### 5. Product Recommendation System

An item-based collaborative filtering recommendation engine was developed.

The system:

1. Creates a customer-product interaction matrix.
2. Uses cosine similarity to identify related products.
3. Finds products similar to items in a customer's purchase history.
4. Generates personalised product recommendations.
5. Excludes products already purchased by the customer.

Recommendation scores represent accumulated similarity scores, not purchase probabilities.

### 6. Model Evaluation

#### Recommendation System Evaluation

A chronological train-test split was used to reduce data leakage.

- **Training period:** 1 December 2010 – 8 November 2011
- **Testing period:** 9 November 2011 – 9 December 2011
- **Evaluation sample:** 300 eligible customers

The collaborative filtering model was compared against a popularity-based baseline.

| Metric | Collaborative Filtering | Popularity Baseline |
|---|---:|---:|
| Precision@5 | 6.67% | 3.20% |
| Recall@5 | 2.95% | 1.07% |
| Catalogue Coverage | 11.81% | 0.80% |

Collaborative filtering outperformed the popularity baseline across all three evaluation metrics.

However, the absolute precision and recall values remain relatively low, indicating opportunities for further improvement.

#### K-Means Clustering Evaluation

| Evaluation Metric | Result |
|---|---:|
| Number of Clusters | 4 |
| Silhouette Score | 0.3375 |
| Davies-Bouldin Index | 1.0086 |
| Calinski-Harabasz Score | 3,328.34 |

Four clusters were retained for business interpretability, although the two-cluster solution achieved a higher silhouette score of approximately 0.433.

## Streamlit Dashboard

An interactive dashboard was developed using Streamlit.

**Dashboard features:**

- Overall customer and product statistics.
- Customer ID-based search.
- Individual customer RFM profiles.
- Customer segment identification.
- Personalised product recommendations.
- Recent purchase history.
- Customer segment distribution.
- Revenue contribution by segment.
- Average spending, purchase frequency, and recency analysis.
- Recommendation model performance comparison.
- K-Means evaluation metrics.

## Project Structure

```text
ai-customer-segmentation-recommendation/
│
├── dataset/
│   └── Online Retail.xlsx
│
├── notebook/
│   └── customer_segmentation.ipynb
│
├── models/
│   └── customer_recommendation_artifacts.joblib
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The dataset directory is required for running the notebook but is not included in the GitHub repository. The saved model artifact is included for running the Streamlit dashboard without rebuilding the models.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/rahulnongmeikapam/ai-customer-segmentation-recommendation.git
cd ai-customer-segmentation-recommendation
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard will usually be available at:

```text
http://localhost:8501
```

### Dataset Setup for Notebook Execution

To run the Jupyter Notebook from the beginning:

1. Download the Online Retail dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail).
2. Create a folder named `dataset` in the project root.
3. Place the downloaded Excel file inside it.
4. Ensure the filename is `Online Retail.xlsx`.

The expected path is:

```text
dataset/Online Retail.xlsx
```

The pre-generated model artifact is included separately in the `models/` directory for dashboard execution.

## Limitations

- Recommendation precision and recall are relatively low and require further improvement.
- Offline recommendation evaluation uses a sample of 300 eligible customers and a limited historical test period.
- Recommendations are based on purchasing similarities rather than product descriptions or semantic understanding.
- K-Means cluster assignments depend on feature selection and the chosen number of clusters.
- Historical purchasing behaviour does not guarantee future customer behaviour.
- Evaluation results do not establish real-world business impact.

## Future Improvements

- Experiment with alternative recommendation algorithms.
- Improve recommendation ranking and personalisation.
- Explore DBSCAN and alternative clustering approaches.
- Incorporate product descriptions into recommendations.
- Improve model evaluation using additional time periods.
- Enhance dashboard design and deployment.

## Author

**Rahul Nongmeikapam**

B.Tech – Computer Science and Engineering (Artificial Intelligence)

GitHub: https://github.com/rahulnongmeikapam

## Acknowledgements

Thanks to the UCI Machine Learning Repository for providing the Online Retail dataset used in this project.
