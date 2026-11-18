# 🛍️ Smart Retail Intelligence System

Turning customer shopping data into meaningful business insights using Python, Machine Learning, and Streamlit.

## 📌 About the Project

I built this project to explore how customer shopping data can be used to understand buying behaviour and support better business decisions.

The "Smart Retail Intelligence System" takes customer shopping data, performs data cleaning and exploratory data analysis, generates sales-related reports, and applies machine learning techniques to group customers based on their shopping patterns.

I also developed an interactive dashboard using Streamlit so that the analysis and results can be explored without having to go through the Python code.

The main idea behind this project is simple: go beyond just looking at numbers and find useful patterns that can help businesses better understand their customers.

## 🎯 Project Objectives

- Clean and prepare raw customer shopping data for analysis.
- Explore customer demographics and shopping behaviour through EDA.
- Identify patterns in product purchases and customer preferences.
- Generate sales analytics reports across different categories.
- Segment customers using machine learning clustering techniques.
- Present the results through an interactive dashboard.

## ✨ Key Features

### 1. Data Ingestion and Cleaning
- Loads the raw customer shopping dataset.
- Handles missing values using suitable imputation methods.
- Removes duplicate records.
- Saves the cleaned dataset for further analysis.

### 2. Exploratory Data Analysis (EDA)
- Dataset overview and statistical summaries.
- Missing value analysis.
- Univariate and categorical analysis.
- Correlation analysis.
- Box plots and visualizations to understand data distributions.

### 3. Sales Analytics
Generates reports to explore:
- Category-wise purchase analysis.
- Product-wise purchase patterns.
- Seasonal shopping behaviour.
- Discount and promotional code usage.
- Gender-wise shopping patterns.
- Payment method preferences.

### 4. Customer Segmentation
- Prepares numerical and categorical features for clustering.
- Applies feature scaling and encoding.
- Uses K-Means clustering to group customers with similar shopping patterns.
- Evaluates clustering using the Elbow Method and Silhouette Score.
- Explores Hierarchical Clustering and DBSCAN for comparison.

### 5. Interactive Streamlit Dashboard
The dashboard brings the different parts of the project together.

Users can:
- View important dataset statistics.
- Explore EDA visualizations.
- Browse sales analytics reports.
- View customer segment distributions.
- Filter and inspect customer records.
- Download generated reports.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation and analysis |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Data preprocessing and clustering |
| SciPy | Hierarchical clustering analysis |
| Streamlit | Interactive dashboard |
| Git & GitHub | Version control and project hosting |
| Docker | Application containerization |

## 📂 Project Structure

```text
Smart-Retail-Intelligence-System/
│
├── app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
│
├── artifacts/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── models/
│   └── visuals/
│
├── logs/
│
└── src/
    └── smart_retail_intelligence/
        ├── __init__.py
        ├── data_ingestion.py
        ├── data_eda.py
        ├── logger.py
        ├── sales_analytics.py
        ├── model_trainer.py
        └── model_pipeline.py
```

## 📊 Dataset

This project uses the **Customer Shopping Trends Dataset** from Kaggle.

Dataset: [Customer Shopping Trends – Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/customer-shopping-trends-dataset)

It contains customer-level shopping information such as age, gender, purchased items, product categories, purchase amounts, location, season, discounts, payment methods, and previous purchases.

## ⚙️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Sanke20/Smart-Retail-Intelligence-System.git
```

### 2. Navigate to the project directory

```bash
cd Smart-Retail-Intelligence-System
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the data pipeline

From the `src` directory, execute:

```bash
python -m smart_retail_intelligence.model_pipeline
```

This runs the data ingestion, EDA, sales analytics, and customer segmentation workflow.

### 5. Launch the Streamlit dashboard

From the project root directory:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal (usually `http://localhost:8501`).

## 🐳 Run Using Docker

You can also run the dashboard inside a Docker container.

### Build the Docker image

```bash
docker build -t smart_retail_intelligence .
```

### Run the container

```bash
docker run -p 8501:8501 smart_retail_intelligence
```

Then open `http://localhost:8501` in your browser.

## 💡 What I Learned

Working on this project helped me gain practical experience in:

- Building a modular Python data science project.
- Cleaning and preparing real-world-style datasets.
- Performing exploratory data analysis.
- Creating reusable analytics functions and reports.
- Applying machine learning clustering algorithms.
- Building interactive dashboards with Streamlit.
- Managing project versions using Git and GitHub.
- Containerizing an application using Docker.

## 🚀 Future Improvements

There is still room to extend this project. Some ideas I would like to explore include:

- Adding more advanced customer segmentation techniques.
- Improving the dashboard with additional interactive filters.
- Integrating a database for storing customer data.
- Adding new analytical features and business insights.
- Exploring deployment options for making the dashboard publicly accessible.

---

**Built as a hands-on data analytics and machine learning project, with a focus on understanding customer behaviour and turning data into actionable insights.**
