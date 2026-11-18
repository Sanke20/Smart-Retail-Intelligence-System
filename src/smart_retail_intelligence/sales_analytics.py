
# src/sales_analysis.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from smart_retail_intelligence.logger import logger


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

FIGURES_DIR = BASE_DIR / "artifacts" / "visuals"

SALES_DATA_PATH = BASE_DIR / "artifacts" / "data" / "processed"

AMOUNT = "Purchase Amount (USD)"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SALES_DATA_PATH, exist_ok=True)


# ============================================================
# 1. SALES KPIs
# ============================================================

def calculate_sales_kpis(df):

    logger.info("Calculating sales KPIs")

    kpis = {
        "customer_records": len(df),
        "unique_customers": df["Customer ID"].nunique(),
        "total_recorded_purchase_amount": df[AMOUNT].sum(),
        "average_purchase_amount": df[AMOUNT].mean(),
        "median_purchase_amount": df[AMOUNT].median(),
        "average_review_rating": df["Review Rating"].mean()
    }

    return kpis


# ============================================================
# 2. CATEGORY ANALYSIS
# ============================================================

def category_analysis(df):

    logger.info("Performing category analysis")

    result = (
        df.groupby("Category")
        .agg(
            customer_records=("Customer ID", "count"),
            total_purchase_amount=(AMOUNT, "sum"),
            average_purchase_amount=(AMOUNT, "mean")
        )
        .reset_index()
    )

    result = result.sort_values(
        "total_purchase_amount",
        ascending=False
    )

    return result


# ============================================================
# 3. PRODUCT ANALYSIS
# ============================================================

def product_analysis(df):

    logger.info("Performing product analysis")

    result = (
        df.groupby("Item Purchased")
        .agg(
            customer_records=("Customer ID", "count"),
            total_purchase_amount=(AMOUNT, "sum"),
            average_purchase_amount=(AMOUNT, "mean")
        )
        .reset_index()
    )

    return result.sort_values(
        "total_purchase_amount",
        ascending=False
    )


# ============================================================
# 4. SEASONAL ANALYSIS
# ============================================================

def seasonal_analysis(df):

    logger.info("Performing seasonal analysis")

    return (
        df.groupby("Season")
        .agg(
            customer_records=("Customer ID", "count"),
            total_purchase_amount=(AMOUNT, "sum"),
            average_purchase_amount=(AMOUNT, "mean")
        )
        .reset_index()
    )


# ============================================================
# 5. DISCOUNT ANALYSIS
# ============================================================

def discount_analysis(df):

    logger.info("Performing discount analysis")

    return (
        df.groupby("Discount Applied")
        .agg(
            customer_records=("Customer ID", "count"),
            average_purchase_amount=(AMOUNT, "mean"),
            total_purchase_amount=(AMOUNT, "sum")
        )
        .reset_index()
    )


# ============================================================
# 6. GENDER ANALYSIS
# ============================================================

def gender_analysis(df):

    logger.info("Performing gender analysis")

    return (
        df.groupby("Gender")
        .agg(
            customer_records=("Customer ID", "count"),
            average_purchase_amount=(AMOUNT, "mean"),
            total_purchase_amount=(AMOUNT, "sum")
        )
        .reset_index()
    )


# ============================================================
# 7. PAYMENT ANALYSIS
# ============================================================

def payment_analysis(df):

    logger.info("Performing payment analysis")

    return (
        df.groupby("Payment Method")
        .agg(
            customer_records=("Customer ID", "count"),
            total_purchase_amount=(AMOUNT, "sum")
        )
        .reset_index()
    )


# ============================================================
# 8. GENERATE SALES REPORTS
# ============================================================

def generate_sales_reports(df):

    logger.info("Generating sales analytics reports")

    category = category_analysis(df)

    products = product_analysis(df)

    seasons = seasonal_analysis(df)

    discounts = discount_analysis(df)

    genders = gender_analysis(df)

    payments = payment_analysis(df)

    print("Saving sales analytics reports...")
    print("Output directory:", SALES_DATA_PATH.resolve())
    category.to_csv(
        SALES_DATA_PATH / "category_sales.csv",
        index=False
    )

    products.to_csv(
        SALES_DATA_PATH / "product_sales.csv",
        index=False
    )

    seasons.to_csv(
        SALES_DATA_PATH / "seasonal_sales.csv",
        index=False
    )

    discounts.to_csv(
        SALES_DATA_PATH / "discount_analysis.csv",
        index=False
    )

    genders.to_csv(
        SALES_DATA_PATH / "gender_sales.csv",
        index=False
    )

    payments.to_csv(
        SALES_DATA_PATH / "payment_analysis.csv",
        index=False
    )
    print("Sales analytics CSV files saved successfully!")
    print("Generated files:")

    for file in SALES_DATA_PATH.glob("*.csv"):
        print(file.name)

    print("Sales analytics reports generated!")

    logger.info("Sales analytics reports generated successfully.")

    return {
        "category": category,
        "products": products,
        "seasons": seasons,
        "discounts": discounts,
        "genders": genders,
        "payments": payments
    }
