# AI-Powered Customer Segmentation & Product Recommendation System

An end-to-end Machine Learning project that analyses customer purchasing behaviour, segments customers using K-Means clustering, and generates personalised product recommendations using item-based collaborative filtering.

## Project Overview

Businesses need to understand their customers to improve customer engagement, retention, and sales.

This project uses historical retail transaction data to:

- Analyse customer purchasing behaviour.
- Identify customer segments using RFM analysis and K-Means clustering.
- Recommend relevant products based on purchasing similarities.
- Compare recommendation performance against a popularity-based baseline.
- Present customer insights through an interactive Streamlit dashboard.

## Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualisation:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn
- **Recommendation System:** Item-Based Collaborative Filtering
- **Clustering:** K-Means
- **Dashboard:** Streamlit
- **Model Storage:** Joblib
- **Development Environment:** Jupyter Notebook, VS Code

## Dataset

**Online Retail Dataset – UCI Machine Learning Repository**

Source: https://archive.ics.uci.edu/dataset/352/online+retail

The dataset contains transactions from a UK-based online retail store between December 2010 and December 2011.

Original dataset:
- 541,909 transaction records
- 8 columns
- 4,372 unique customers

After data cleaning:
- 392,692 transaction records
- 4,338 unique customers
- 3,665 unique products

## Project Workflow

### 1. Data Cleaning

Performed:
- Duplicate removal.
- Cancellation transaction removal.
- Missing Customer ID removal.
- Invalid quantity and price filtering.
- Total transaction value calculation.

### 2. Exploratory Data Analysis

Analysed:
- Customer purchasing behaviour.
- Top-selling products.
- Revenue by country.
- Monthly revenue trends.
- Highest-spending customers.

### 3. RFM Analysis

Created three customer features:

- **Recency:** Days since the customer's latest purchase.
- **Frequency:** Number of distinct invoices.
- **Monetary:** Total customer spending.

Applied log transformation and StandardScaler to prepare the features for clustering.

### 4. Customer Segmentation Using K-Means

Implemented K-Means clustering with four customer segments:

| Segment | Customers | Percentage |
|---|---:|---:|
| Champions | 713 | 16.44% |
| At-Risk Customers | 1,622 | 37.39% |
| Recent Customers | 837 | 19.29% |
| Regular Customers | 1,166 | 26.88% |

These segments support different business strategies, including loyalty rewards, customer re-engagement, repeat-purchase campaigns, and complementary product promotions.

### 5. Product Recommendation System

Developed an item-based collaborative filtering recommendation engine.

The system:
- Creates a customer-product interaction matrix.
- Uses cosine similarity to identify related products.
- Generates recommendations based on a customer's purchase history.
- Excludes products already purchased by the customer.

Recommendation scores represent accumulated similarity scores, not purchase probabilities.

### 6. Model Evaluation

#### Recommendation System Evaluation

Used a chronological train-test split, with 9 November 2011 as the split date.

Evaluated the collaborative filtering model against a popularity-based baseline using a sample of 300 eligible customers.

| Metric | Collaborative Filtering | Popularity Baseline |
|---|---:|---:|
| Precision@5 | 6.67% | 3.20% |
| Recall@5 | 2.95% | 1.07% |
| Catalogue Coverage | 11.81% | 0.80% |

Collaborative filtering outperformed the popularity baseline across all three metrics. However, the absolute recommendation scores remain low, leaving room for improvement.

#### K-Means Evaluation

| Metric | Result |
|---|---:|
| Number of Clusters | 4 |
| Silhouette Score | 0.3375 |
| Davies-Bouldin Index | 1.0086 |
| Calinski-Harabasz Score | 3328.34 |

Four clusters were retained for business interpretability, although the two-cluster solution achieved a higher silhouette score.

## Streamlit Dashboard

Developed an interactive dashboard featuring:

- Overall customer and product statistics.
- Customer ID-based search.
- Individual customer RFM profiles.
- Customer segment identification.
- Personalised product recommendations.
- Recent purchase history.
- Customer segment distribution charts.
- Revenue contribution analysis.
- Model performance comparison.

## Project Structure

```text
AI Customer Segmentation/
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

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Customer-Segmentation
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

The dashboard will open in your browser, usually at:

```text
http://localhost:8501
```

### Dataset Setup

The original dataset is not included in the repository. Download it from the UCI Machine Learning Repository and place the Excel file at:

```text
dataset/Online Retail.xlsx
```

The saved model artifact is included separately in the `models/` directory for dashboard execution.

## Limitations

- Recommendation accuracy is relatively low and requires further improvement.
- Offline evaluation uses a sample of 300 eligible customers and a limited historical test period.
- Recommendations are based on purchasing similarities rather than product descriptions or semantic understanding.
- K-Means cluster assignments depend on the selected features and number of clusters.
- Historical customer behaviour does not guarantee future purchasing behaviour.

## Future Improvements

- Experiment with alternative recommendation algorithms.
- Improve recommendation ranking and personalisation.
- Explore DBSCAN and alternative clustering approaches.
- Incorporate product descriptions into recommendations.
- Improve dashboard design and deployment.
- Evaluate models on additional time periods.

## Author

**Rahul Nongmeikapam**

B.Tech – Computer Science and Engineering (Artificial Intelligence)

## Acknowledgements

UCI Machine Learning Repository for providing the Online Retail dataset.
