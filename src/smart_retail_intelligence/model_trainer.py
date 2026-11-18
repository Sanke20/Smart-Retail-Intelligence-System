
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

from smart_retail_intelligence.logger import logger


# ============================================================
# MODEL TRAINER CLASS
# ============================================================

class ModelTrainer:

    def __init__(
        self,
        model_path,
        visual_path
    ):

        self.model_path = model_path
        self.visual_path = visual_path

        os.makedirs(
            self.model_path,
            exist_ok=True
        )

        os.makedirs(
            self.visual_path,
            exist_ok=True
        )

    # ---------------------------------------------------------
    # Prepare Features
    # ---------------------------------------------------------

    def prepare_features(
        self,
        df,
        clustering_features
    ):

        logger.info(
            "Preparing clustering features."
        )

        # -----------------------------------------------------
        # Select clustering features
        # -----------------------------------------------------

        clustering_features = [
            col for col in clustering_features
            if col in df.columns
            and "id" not in col.lower()
        ]

        if not clustering_features:
            raise ValueError(
                "No valid clustering features found."
            )

        X = df[
            clustering_features
        ].copy()

        # -----------------------------------------------------
        # Separate Numerical and Categorical Features
        # -----------------------------------------------------

        numerical_features = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        if not numerical_features and not categorical_features:
            raise ValueError(
                "No numerical or categorical features available."
            )

        logger.info(
            f"Numerical Features: {numerical_features}"
        )

        logger.info(
            f"Categorical Features: {categorical_features}"
        )

        # -----------------------------------------------------
        # Numerical Feature Preprocessing
        # -----------------------------------------------------

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        # -----------------------------------------------------
        # Categorical Feature Preprocessing
        # -----------------------------------------------------

        # Support newer and older scikit-learn versions

        try:

            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )

        except TypeError:

            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False
            )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    encoder
                )
            ]
        )

        # -----------------------------------------------------
        # Combine Numerical and Categorical Pipelines
        # -----------------------------------------------------

        transformers = []

        if numerical_features:

            transformers.append(
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_features
                )
            )

        if categorical_features:

            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        # -----------------------------------------------------
        # Apply Preprocessing
        # -----------------------------------------------------

        logger.info(
            "Applying feature imputation, encoding and scaling."
        )

        X_transformed = preprocessor.fit_transform(
            X
        )

        X_scaled = np.asarray(
            X_transformed,
            dtype=np.float64
        )

        logger.info(
            "Feature preprocessing completed."
        )

        print("\nOriginal Feature Shape:", X.shape)

        print(
            "Transformed Feature Shape:",
            X_scaled.shape
        )

        return (
            X_scaled,
            preprocessor
        )

    # ---------------------------------------------------------
    # Elbow Method
    # ---------------------------------------------------------

    def elbow_analysis(
        self,
        X_scaled,
        k_range=range(2, 11)
    ):

        logger.info(
            "Running K-Means elbow analysis."
        )

        inertia_values = []

        k_values = list(k_range)

        for k in k_values:

            if k >= len(X_scaled):
                logger.warning(
                    f"Skipping K={k}: insufficient customer records."
                )
                continue

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            model.fit(
                X_scaled
            )

            inertia_values.append(
                model.inertia_
            )

        valid_k_values = k_values[:len(inertia_values)]

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            valid_k_values,
            inertia_values,
            marker="o"
        )

        plt.xlabel(
            "Number of Clusters (K)"
        )

        plt.ylabel(
            "Inertia"
        )

        plt.title(
            "Elbow Method - Customer Segmentation"
        )

        plt.xticks(valid_k_values)

        plt.grid(
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.visual_path,
            "elbow_plot.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        logger.info(
            f"Elbow plot saved at {output_path}"
        )

        return inertia_values

    # ---------------------------------------------------------
    # Silhouette Analysis
    # ---------------------------------------------------------

    def silhouette_analysis(
        self,
        X_scaled,
        k_range=range(2, 11)
    ):

        logger.info(
            "Running silhouette analysis."
        )

        silhouette_scores = {}

        for k in k_range:

            if k >= len(X_scaled):
                logger.warning(
                    f"Skipping K={k}: insufficient customer records."
                )
                continue

            model = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(
                X_scaled
            )

            score = silhouette_score(
                X_scaled,
                labels,
                sample_size=min(2000, len(X_scaled)),
                random_state=42
)
            

            silhouette_scores[k] = score

            logger.info(
                f"K={k}, Silhouette Score={score:.4f}"
            )

        if not silhouette_scores:

            raise ValueError(
                "Unable to calculate silhouette scores."
            )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            list(silhouette_scores.keys()),
            list(silhouette_scores.values()),
            marker="o"
        )

        plt.xlabel(
            "Number of Clusters (K)"
        )

        plt.ylabel(
            "Silhouette Score"
        )

        plt.title(
            "Silhouette Score Analysis - Customer Segmentation"
        )

        plt.grid(
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.visual_path,
            "silhouette_score.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        logger.info(
            f"Silhouette plot saved at {output_path}"
        )

        return silhouette_scores

    # ---------------------------------------------------------
    # Hierarchical Clustering
    # ---------------------------------------------------------

    def hierarchical_clustering(
        self,
        X_scaled,
        max_samples=1000
    ):

        logger.info(
            "Running hierarchical clustering."
        )

        # -----------------------------------------------------
        # Limit the number of records for dendrogram generation
        # to reduce memory and execution time.
        # -----------------------------------------------------

        if len(X_scaled) > max_samples:

            logger.warning(
                f"Using a sample of {max_samples} customers "
                "for dendrogram visualization."
            )

            rng = np.random.default_rng(42)

            sample_indices = rng.choice(
                len(X_scaled),
                size=max_samples,
                replace=False
            )

            X_sample = X_scaled[sample_indices]

        else:

            X_sample = X_scaled

        linkage_matrix = linkage(
            X_sample,
            method="ward"
        )

        plt.figure(
            figsize=(15, 7)
        )

        dendrogram(
            linkage_matrix,
            truncate_mode="level",
            p=5
        )

        plt.title(
            "Hierarchical Clustering Dendrogram - Retail Customers"
        )

        plt.xlabel(
            "Customers"
        )

        plt.ylabel(
            "Distance"
        )

        plt.tight_layout()

        output_path = os.path.join(
            self.visual_path,
            "hierarchical_dendrogram.png"
        )

        plt.savefig(
            output_path,
            dpi=300
        )

        plt.close()

        logger.info(
            f"Dendrogram saved at {output_path}"
        )

        return linkage_matrix

    # ---------------------------------------------------------
    # DBSCAN
    # ---------------------------------------------------------

    
    # ---------------------------------------------------------
    # DBSCAN
    # ---------------------------------------------------------

    def dbscan_analysis(
        self,
        X_scaled,
        eps=1.5,
        min_samples=10
    ):

        logger.info(
            "Running DBSCAN."
        )

        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = dbscan.fit_predict(
            X_scaled
        )

        unique_labels = set(
            labels
        )

        logger.info(
            f"DBSCAN clusters: {unique_labels}"
        )

        # -----------------------------------------------------
        # DBSCAN Evaluation - Exclude Noise Points
        # -----------------------------------------------------

        non_noise_mask = labels != -1

        non_noise_labels = labels[non_noise_mask]

        n_clusters = len(
            set(non_noise_labels)
        )

        if (
            n_clusters >= 2
            and n_clusters < len(non_noise_labels)
        ):

            score = silhouette_score(
                X_scaled[non_noise_mask],
                non_noise_labels
            )

            logger.info(
                f"DBSCAN Silhouette Score: {score:.4f}"
            )

        else:

            logger.warning(
                "Silhouette Score cannot be calculated "
                "for the current DBSCAN result."
            )

        return labels


        

    # ---------------------------------------------------------
    # Final K-Means
    # ---------------------------------------------------------

    def train_final_model(
        self,
        X_scaled,
        optimal_k
    ):

        logger.info(
            f"Training final K-Means model with K={optimal_k}"
        )

        final_model = KMeans(
            n_clusters=optimal_k,
            random_state=42,
            n_init=10
        )

        cluster_labels = final_model.fit_predict(
            X_scaled
        )

        # -----------------------------------------------------
        # Save model
        # -----------------------------------------------------

        model_file = os.path.join(
            self.model_path,
            "kmeans_model.joblib"
        )

        joblib.dump(
            final_model,
            model_file
        )

        logger.info(
            f"K-Means model saved at {model_file}"
        )

        return (
            final_model,
            cluster_labels
        )

    # ---------------------------------------------------------
    # Save Preprocessing Objects
    # ---------------------------------------------------------

    def save_preprocessor(
        self,
        preprocessor
    ):

        preprocessor_path = os.path.join(
            self.model_path,
            "preprocessor.joblib"
        )

        joblib.dump(
            preprocessor,
            preprocessor_path
        )

        logger.info(
            "Preprocessor saved successfully."
        )

    # ---------------------------------------------------------
    # Cluster Profile
    # ---------------------------------------------------------

    def create_cluster_profile(
        self,
        df,
        cluster_labels,
        clustering_features
    ):

        logger.info(
            "Creating cluster profile."
        )

        profile_df = df.copy()

        profile_df[
            "Cluster"
        ] = cluster_labels

        # -----------------------------------------------------
        # Separate numerical and categorical features
        # -----------------------------------------------------

        numerical_features = profile_df[
            clustering_features
        ].select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_features = profile_df[
            clustering_features
        ].select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()

        # -----------------------------------------------------
        # Numerical Cluster Profile
        # -----------------------------------------------------

        if numerical_features:

            cluster_profile = (
                profile_df
                .groupby("Cluster")[numerical_features]
                .mean()
                .round(2)
            )

            cluster_profile.columns = [
                f"Average_{col}"
                for col in cluster_profile.columns
            ]

        else:

            cluster_profile = pd.DataFrame(
                index=sorted(profile_df["Cluster"].unique())
            )

        # -----------------------------------------------------
        # Categorical Cluster Profile
        # -----------------------------------------------------

        if categorical_features:

            categorical_profile = (
                profile_df
                .groupby("Cluster")[categorical_features]
                .agg(
                    lambda x: (
                        x.mode().iloc[0]
                        if not x.mode().empty
                        else "Unknown"
                    )
                )
            )

            categorical_profile.columns = [
                f"Most_Common_{col}"
                for col in categorical_profile.columns
            ]

            cluster_profile = cluster_profile.join(
                categorical_profile
            )

        # -----------------------------------------------------
        # Customer Count and Percentage
        # -----------------------------------------------------

        cluster_counts = (
            profile_df[
                "Cluster"
            ]
            .value_counts()
            .sort_index()
        )

        cluster_profile[
            "Customer_Count"
        ] = cluster_counts

        cluster_profile[
            "Percentage"
        ] = (
            cluster_counts /
            len(profile_df) *
            100
        ).round(2)

        # -----------------------------------------------------
        # Save Cluster Profile
        # -----------------------------------------------------

        profile_path = os.path.join(
            self.model_path,
            "cluster_profile.csv"
        )

        cluster_profile.to_csv(
            profile_path
        )

        logger.info(
            f"Cluster profile saved at {profile_path}"
        )

        return (
            profile_df,
            cluster_profile
        )