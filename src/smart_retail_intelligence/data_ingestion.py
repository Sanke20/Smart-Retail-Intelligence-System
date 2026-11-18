
import os
import pandas as pd

from smart_retail_intelligence.logger import logger


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(file_path):

    logger.info(f"Loading dataset from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully")
    print("Initial Shape:", df.shape)

    return df


# ============================================================
# 2. CLEAN DATA
# ============================================================

def clean_data(df):

    df = df.copy()

    logger.info("Handling duplicate records")

    initial_rows = len(df)

    df = df.drop_duplicates().copy()

    print(
        f"Removed {initial_rows - len(df)} duplicate rows."
    )

    logger.info("Handling missing values")

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            # Numerical columns: fill with median
            median_value = df[col].median()

            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)

        else:

            # String/Object/Categorical columns: fill with mode
            mode_vals = df[col].mode()

            if not mode_vals.empty:
                df[col] = df[col].fillna(mode_vals[0])
            else:
                df[col] = df[col].fillna("Unknown")

    print("Missing values handled.")

    logger.info("Data cleaning completed successfully.")

    return df


# ============================================================
# 3. DATA INGESTION CLASS
# ============================================================

class DataIngestion:

    def __init__(
        self,
        input_path,
        processed_path
    ):

        self.input_path = input_path
        self.processed_path = processed_path

        os.makedirs(
            os.path.dirname(self.processed_path),
            exist_ok=True
        )

    def initiate_data_ingestion(self):

        logger.info("Starting data ingestion.")

        # Load dataset
        df = load_data(
            self.input_path
        )

        # Clean dataset
        df = clean_data(
            df
        )

        # Save processed dataset
        df.to_csv(
            self.processed_path,
            index=False
        )

        logger.info(
            f"Processed dataset saved at: {self.processed_path}"
        )

        print(
            "\nProcessed Dataset Shape:",
            df.shape
        )

        return df
