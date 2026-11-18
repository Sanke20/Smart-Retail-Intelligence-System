from pathlib import Path
import pandas as pd
import streamlit as st

# Keep this app.py in the capstone_project root folder.
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "artifacts" / "data" / "processed"
VISUALS_DIR = BASE_DIR / "artifacts" / "visuals"

st.set_page_config(page_title="Smart Retail Intelligence", page_icon="🛍️", layout="wide")


@st.cache_data
def load_csv(path_string):
    path = Path(path_string)
    return pd.read_csv(path) if path.exists() else None


def money(value):
    return "N/A" if pd.isna(value) else f"${value:,.2f}"


def missing_file(name):
    st.info(f"`{name}` was not found. Run your model pipeline and check `artifacts/data/processed/`.")


processed = load_csv(str(PROCESSED_DIR / "processed_customer_data.csv"))
segmented = load_csv(str(PROCESSED_DIR / "segmented_customer_data.csv"))

reports = {
    "Category analysis": ("category_sales.csv", load_csv(str(PROCESSED_DIR / "category_sales.csv"))),
    "Product analysis": ("product_sales.csv", load_csv(str(PROCESSED_DIR / "product_sales.csv"))),
    "Seasonal analysis": ("seasonal_sales.csv", load_csv(str(PROCESSED_DIR / "seasonal_sales.csv"))),
    "Discount analysis": ("discount_analysis.csv", load_csv(str(PROCESSED_DIR / "discount_analysis.csv"))),
    "Gender analysis": ("gender_sales.csv", load_csv(str(PROCESSED_DIR / "gender_sales.csv"))),
    "Payment analysis": ("payment_analysis.csv", load_csv(str(PROCESSED_DIR / "payment_analysis.csv"))),
}

st.sidebar.title("🛍️ Smart Retail Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Project Overview", "Exploratory Data Analysis", "Sales Analytics", "Customer Segmentation", "Data Explorer"],
)
st.sidebar.caption("Python • Pandas • Scikit-learn • Streamlit")

st.title("🛍️ Smart Retail Intelligence System")
st.write("Explore customer shopping behaviour, purchase patterns, sales summaries and ML-based customer segments.")

if page == "Project Overview":
    st.header("Project Overview")
    if processed is None:
        missing_file("processed_customer_data.csv")
    else:
        records = len(processed)
        customers = processed["Customer ID"].nunique() if "Customer ID" in processed else records
        amount = processed["Purchase Amount (USD)"] if "Purchase Amount (USD)" in processed else None
        cols = st.columns(4)
        cols[0].metric("Customer records", f"{records:,}")
        cols[1].metric("Unique customers", f"{customers:,}")
        cols[2].metric("Total recorded purchase amount", money(amount.sum()) if amount is not None else "N/A")
        cols[3].metric("Average purchase amount", money(amount.mean()) if amount is not None else "N/A")
        st.subheader("Processed dataset preview")
        st.dataframe(processed.head(10), use_container_width=True)
        st.subheader("Project workflow")
        st.markdown("""
        1. **Data ingestion:** load and clean the shopping trends dataset.
        2. **EDA:** inspect data quality, distributions and relationships.
        3. **Sales analytics:** summarize category, product, season, discount, gender and payment patterns.
        4. **Feature preparation:** prepare numerical and categorical features for clustering.
        5. **Model analysis:** evaluate clustering approaches and select a K-Means model.
        6. **Customer segmentation:** review cluster assignments and customer profiles.
        """)
        st.caption("Dataset note: these are customer-level shopping records. The sum is the total of recorded purchase amounts, not audited company revenue or a monthly revenue trend.")

elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    if processed is None:
        missing_file("processed_customer_data.csv")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{processed.shape[0]:,}")
        c2.metric("Columns", f"{processed.shape[1]:,}")
        c3.metric("Duplicate rows", f"{processed.duplicated().sum():,}")
        st.subheader("Missing values by column")
        missing = processed.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found in the processed dataset.")
        else:
            st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True)
        st.subheader("EDA visualizations")
        if VISUALS_DIR.exists() and list(VISUALS_DIR.glob("*.png")):
            for image_path in sorted(VISUALS_DIR.glob("*.png")):
                st.markdown(f"**{image_path.stem.replace('_', ' ').title()}**")
                st.image(str(image_path), use_container_width=True)
        else:
            st.info("No PNG charts found in artifacts/visuals yet. Run the EDA pipeline.")

elif page == "Sales Analytics":
    st.header("Sales Analytics")
    st.caption("Summaries of purchase records in the customer shopping trends dataset.")
    if processed is not None and "Purchase Amount (USD)" in processed:
        amount = processed["Purchase Amount (USD)"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Total recorded purchase amount", money(amount.sum()))
        c2.metric("Average purchase amount", money(amount.mean()))
        c3.metric("Median purchase amount", money(amount.median()))
    choice = st.selectbox("Choose a report", list(reports.keys()))
    filename, report = reports[choice]
    if report is None:
        missing_file(filename)
    else:
        st.subheader(choice)
        st.dataframe(report, use_container_width=True)
        categories = [col for col in report if report[col].dtype == "object"]
        metrics = [col for col in report if pd.api.types.is_numeric_dtype(report[col])]
        if categories and metrics:
            preferred = [c for c in metrics if "total" in c.lower() or "amount" in c.lower()]
            metric = preferred[0] if preferred else metrics[0]
            st.bar_chart(report.set_index(categories[0])[[metric]])
        st.download_button("Download report CSV", report.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")

elif page == "Customer Segmentation":
    st.header("Customer Segmentation")
    if segmented is None:
        missing_file("segmented_customer_data.csv")
    else:
        cluster_col = next((c for c in ["Cluster", "cluster", "KMeans_Cluster"] if c in segmented.columns), None)
        if cluster_col is None:
            st.warning("Could not find a cluster-label column. Check the columns in segmented_customer_data.csv.")
        else:
            counts = segmented[cluster_col].value_counts().sort_index()
            c1, c2 = st.columns(2)
            c1.metric("Customer records segmented", f"{len(segmented):,}")
            c2.metric("Number of clusters", f"{segmented[cluster_col].nunique():,}")
            st.subheader("Customer count by cluster")
            st.bar_chart(counts)
            cluster = st.selectbox("Explore a cluster", sorted(segmented[cluster_col].dropna().unique(), key=str))
            selected = segmented[segmented[cluster_col] == cluster]
            st.write(f"Records in this cluster: **{len(selected):,}**")
            st.dataframe(selected.head(100), use_container_width=True)
            st.download_button("Download selected cluster", selected.to_csv(index=False).encode("utf-8"), file_name=f"cluster_{cluster}_customers.csv", mime="text/csv")
        profile = load_csv(str(PROCESSED_DIR / "cluster_profile.csv"))
        if profile is not None:
            st.subheader("Cluster profile")
            st.dataframe(profile, use_container_width=True)
            st.download_button("Download cluster profile", profile.to_csv(index=False).encode("utf-8"), file_name="cluster_profile.csv", mime="text/csv")

elif page == "Data Explorer":
    st.header("Data Explorer")
    datasets = {"Processed customer data": processed, "Segmented customer data": segmented}
    datasets.update({name: data for name, (_, data) in reports.items()})
    profile = load_csv(str(PROCESSED_DIR / "cluster_profile.csv"))
    datasets["Cluster profile"] = profile
    available = {name: data for name, data in datasets.items() if data is not None}
    if not available:
        st.warning("No generated CSV files were found. Run your pipeline first.")
    else:
        name = st.selectbox("Select a dataset", list(available))
        data = available[name]
        st.write(f"Shape: **{data.shape[0]:,} rows × {data.shape[1]} columns**")
        st.dataframe(data, use_container_width=True)
        st.download_button("Download selected dataset", data.to_csv(index=False).encode("utf-8"), file_name=f"{name.lower().replace(' ', '_')}.csv", mime="text/csv")

st.divider()
st.caption("Smart Retail Intelligence System | Interactive dashboard")
