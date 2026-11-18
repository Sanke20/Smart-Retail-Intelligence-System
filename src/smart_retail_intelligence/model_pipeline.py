print('MODEL PIPELINE MODULE LOADED')
import os
import pandas as pd

from pathlib import Path

from smart_retail_intelligence.logger import logger
from smart_retail_intelligence.data_ingestion import DataIngestion
from smart_retail_intelligence.data_eda import DataEDA
from smart_retail_intelligence.sales_analytics import generate_sales_reports
from smart_retail_intelligence.model_trainer import ModelTrainer


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DATA_PATH = (
    BASE_DIR / "artifacts" / "data" / "raw" /
    "shopping_trends.csv"
)

PROCESSED_DATA_PATH = (
    BASE_DIR / "artifacts" / "data" / "processed" /
    "processed_customer_data.csv"
)

VISUAL_PATH = (
    BASE_DIR / "artifacts" / "visuals"
)

MODEL_PATH = (
    BASE_DIR / "artifacts" / "models"
)

SEGMENTED_DATA_PATH = (
    BASE_DIR / "artifacts" / "data" / "processed" /
    "segmented_customer_data.csv"
)

# Ensure directories exist

os.makedirs(
    PROCESSED_DATA_PATH.parent,
    exist_ok=True
)

os.makedirs(
    VISUAL_PATH,
    exist_ok=True
)

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)


# ============================================================
# CLUSTERING FEATURES
# ============================================================

CLUSTERING_FEATURES = [

    "Age",

    "Purchase Amount (USD)",

    "Review Rating",

    "Previous Purchases",

    "Gender",

    "Item Purchased",

    "Category",

    "Location",

    "Size",

    "Color",

    "Season",

    "Subscription Status",

    "Shipping Type",

    "Discount Applied",

    "Promo Code Used",

    "Payment Method",

    "Frequency of Purchases"
]


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    try:

        logger.info(
            "=================================================="
        )

        logger.info(
            "Smart Retail Customer Segmentation Pipeline Started"
        )

        logger.info(
            "=================================================="
        )


        # ----------------------------------------------------
        # STEP 1: DATA INGESTION
        # ----------------------------------------------------

        logger.info("STEP 1: Starting data ingestion.")

        data_ingestion = DataIngestion(
            input_path=str(RAW_DATA_PATH),
            processed_path=str(PROCESSED_DATA_PATH)
        )

        df = data_ingestion.initiate_data_ingestion()

        logger.info(
            f"Data ingestion completed. Shape: {df.shape}"
        )


        # ----------------------------------------------------
        # VALIDATE REQUIRED COLUMNS
        # ----------------------------------------------------

        logger.info(
            "Validating required dataset columns."
        )

        missing_columns = [
            col for col in CLUSTERING_FEATURES
            if col not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing required clustering columns: "
                f"{missing_columns}"
            )

        if "Customer ID" not in df.columns:

            raise ValueError(
                "Customer ID column is missing from the dataset."
            )

        if df.empty:

            raise ValueError(
                "The processed dataset is empty."
            )

        logger.info(
            "Required dataset columns validated successfully."
        )


        # ----------------------------------------------------
        # STEP 2: EDA
        # ----------------------------------------------------

        logger.info("STEP 2: Starting EDA.")

        
# ----------------------------------------------------
# STEP 2: EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------

        logger.info("STEP 2: Starting EDA.")

        print("\nSTEP 2: Starting Exploratory Data Analysis...")

        # Initialize EDA class
        eda = DataEDA(
            visual_path=str(VISUAL_PATH)

        )

        # Dataset overview
        eda.dataset_overview(df)

        # Missing value analysis
        eda.missing_value_analysis(df)

        # Identify numerical and categorical columns
        numerical_columns, categorical_columns = eda.identify_columns(df)

        # Exclude Customer ID from EDA analysis
        numerical_columns = [
            col for col in numerical_columns
            if col != "Customer ID"
        ]

        categorical_columns = [
            col for col in categorical_columns
            if col != "Customer ID"
        ]

        # Numerical summary
        if numerical_columns:
            eda.numerical_summary(
                df,
                numerical_columns
            )

            # Correlation analysis requires at least two numerical columns
            if len(numerical_columns) >= 2:
                eda.correlation_analysis(
                    df,
                    numerical_columns
                )

            # Distribution plots
            eda.univariate_analysis(
                df,
                numerical_columns
            )

            # Boxplots
            eda.boxplot_analysis(
                df,
                numerical_columns
            )

        # Categorical analysis
        if categorical_columns:
            eda.categorical_analysis(
                df,
                categorical_columns
            )

        logger.info("EDA completed successfully.")

        print("\nEDA completed successfully.")


        # ----------------------------------------------------
        # STEP 3: SALES ANALYTICS
        # ----------------------------------------------------

        logger.info(
            "STEP 3: Starting sales analytics."
        )

        sales_reports = generate_sales_reports(
            df
        )

        logger.info(
            "Sales analytics completed successfully."
        )

        print("\nSales Analytics Reports Generated Successfully.")


        # ----------------------------------------------------
        # STEP 4: MODEL TRAINER
        # ----------------------------------------------------

        logger.info(
            "STEP 4: Preparing clustering model."
        )

        model_trainer = ModelTrainer(
            model_path=str(MODEL_PATH),
            visual_path=str(VISUAL_PATH)
        )


        # ----------------------------------------------------
        # STEP 5: FEATURE PREPARATION
        # ----------------------------------------------------

        logger.info(
            "STEP 5: Preparing clustering features."
        )

        (
            X_scaled,
            preprocessor
        ) = model_trainer.prepare_features(
            df,
            CLUSTERING_FEATURES
        )


        # ----------------------------------------------------
        # STEP 6: ELBOW ANALYSIS
        # ----------------------------------------------------

        logger.info(
            "STEP 6: Running elbow analysis."
        )

        model_trainer.elbow_analysis(
            X_scaled,
            k_range=range(2, 9)
        )


        # ----------------------------------------------------
        # STEP 7: SILHOUETTE ANALYSIS
        # ----------------------------------------------------

        logger.info(
            "STEP 7: Running silhouette analysis."
        )

        silhouette_scores = (
            model_trainer.silhouette_analysis(
                X_scaled,
                k_range=range(2, 9)
            )
        )

        print("\nSilhouette Scores:")

        for k, score in silhouette_scores.items():

            print(
                f"K = {k} --> "
                f"Silhouette Score = {score:.4f}"
            )


        # ----------------------------------------------------
        # STEP 8: SELECT OPTIMAL K
        # ----------------------------------------------------

        optimal_k = max(
            silhouette_scores,
            key=silhouette_scores.get
        )

        logger.info(
            f"Optimal K : {optimal_k}"
        )

        print(
            f"\nOptimal K: {optimal_k}"
        )


        # ----------------------------------------------------
        # STEP 9: HIERARCHICAL CLUSTERING
        # ----------------------------------------------------

        logger.info(
            "STEP 9: Running hierarchical clustering."
        )

        model_trainer.hierarchical_clustering(
            X_scaled
        )


        # ----------------------------------------------------
        # STEP 10: DBSCAN
        # ----------------------------------------------------

        logger.info(
            "STEP 10: Running DBSCAN."
        )

        dbscan_labels = (
            model_trainer.dbscan_analysis(
                X_scaled
            )
        )

        print("\nDBSCAN Cluster Distribution:")

        print(
            pd.Series(dbscan_labels)
            .value_counts()
            .sort_index()
        )


        # ----------------------------------------------------
        # STEP 11: FINAL K-MEANS
        # ----------------------------------------------------

        logger.info(
            "STEP 11: Training final K-Means model."
        )

        (
            final_model,
            cluster_labels
        ) = model_trainer.train_final_model(
            X_scaled,
            optimal_k
        )


        # ----------------------------------------------------
        # STEP 12: SAVE PREPROCESSING OBJECTS
        # ----------------------------------------------------

        logger.info(
            "STEP 12: Saving preprocessing objects."
        )

        model_trainer.save_preprocessor(
            preprocessor
        )


        # ----------------------------------------------------
        # STEP 13: CLUSTER PROFILE
        # ----------------------------------------------------

        logger.info(
            "STEP 13: Creating cluster profiles."
        )

        (
            clustered_df,
            cluster_profile
        ) = model_trainer.create_cluster_profile(
            df,
            cluster_labels,
            CLUSTERING_FEATURES
        )


        # ----------------------------------------------------
        # SAVE DBSCAN CLUSTER LABELS
        # ----------------------------------------------------

        clustered_df[
            "DBSCAN_Cluster"
        ] = dbscan_labels


        # ----------------------------------------------------
        # SAVE FINAL SEGMENTED DATA
        # ----------------------------------------------------

        clustered_df.to_csv(
            SEGMENTED_DATA_PATH,
            index=False
        )

        logger.info(
            f"Segmented customer data saved at "
            f"{SEGMENTED_DATA_PATH}"
        )


        # ----------------------------------------------------
        # FINAL OUTPUT
        # ----------------------------------------------------

        print("\n==============================================")
        print("CUSTOMER SEGMENTATION COMPLETED")
        print("==============================================")

        print(
            f"\nFinal Number of Clusters: {optimal_k}"
        )

        print(
            "\nCluster Distribution:"
        )

        print(
            clustered_df[
                "Cluster"
            ].value_counts().sort_index()
        )

        print(
            "\nCluster Profile:"
        )

        print(
            cluster_profile
        )

        print(
            "\nSales Analytics Reports:"
        )

        for report_name, report_df in sales_reports.items():

            print(
                f"{report_name}: {report_df.shape}"
            )

        logger.info(
            "Smart Retail Customer Segmentation Pipeline "
            "completed successfully."
        )


    except Exception as e:

        logger.exception(
            f"Pipeline failed: {e}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
