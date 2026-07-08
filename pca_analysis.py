"""
Milestone Assignment 2: Principal Component Analysis
Anderson Cancer Center - Dimensionality Reduction Project

This script demonstrates the full workflow, using concepts from
Modules 1-5:
    - Variables
    - Loops
    - Conditional statements within loops
    - Exception (error) handling
    - Functions
    - Dictionaries
    - Exporting data to CSV
    - Class creation (object-oriented design)
    - Data preparation, cleaning, exploration, manipulation,
      and preprocessing
    - Data visualization
    - PCA dimensionality reduction
    - (Bonus) Logistic Regression prediction

Author: Sodiq Omoniyi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Module-level variables (constants/config) — demonstrates "variables"
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_COMPONENTS = 2
VARIANCE_ALERT_THRESHOLD = 0.05  # flag components explaining < 5% variance

# A dictionary used throughout the script to keep track of run metadata
# and results — demonstrates "dictionary" usage.
RESULTS_SUMMARY = {
    "n_samples": None,
    "n_features": None,
    "missing_values_found": None,
    "duplicate_rows_found": None,
    "explained_variance": {},
    "logistic_regression_accuracy": None,
}


class CancerPCAAnalyzer:
    """
    A class that encapsulates the entire PCA workflow for the
    Breast Cancer Wisconsin dataset: data preparation, cleaning,
    exploration, PCA dimensionality reduction, visualization, and
    an optional logistic regression classifier.

    Demonstrates class creation / object-oriented programming.
    """

    def __init__(self, n_components=N_COMPONENTS, random_state=RANDOM_STATE):
        self.n_components = n_components
        self.random_state = random_state
        self.features = None
        self.target = None
        self.target_names = None
        self.scaled_features = None
        self.pca = None
        self.principal_components = None
        self.model = None

    # ------------------------------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------------------------------
    def load_data(self):
        """
        Load the breast cancer dataset from sklearn.datasets.
        Wrapped in a try/except block to demonstrate exception handling,
        in case the dataset fails to load (e.g. corrupted install).
        """
        try:
            data = load_breast_cancer()
            self.features = pd.DataFrame(data.data, columns=data.feature_names)
            self.target = pd.Series(data.target, name="diagnosis")
            self.target_names = data.target_names
            print("Data loaded successfully.")
        except Exception as error:
            # Exception handling: if loading fails, raise a clear message
            # instead of letting the program crash with a cryptic trace.
            raise RuntimeError(f"Failed to load dataset: {error}") from error

        RESULTS_SUMMARY["n_samples"] = self.features.shape[0]
        RESULTS_SUMMARY["n_features"] = self.features.shape[1]
        return self.features, self.target

    # ------------------------------------------------------------------
    # DATA EXPLORATION
    # ------------------------------------------------------------------
    def explore_data(self):
        """
        Explore the dataset: shape, summary statistics, class balance,
        and a check for missing/duplicate values.
        Uses a loop with a conditional statement inside it to inspect
        each column for missing values.
        """
        print("\n--- Data Exploration ---")
        print(f"Shape: {self.features.shape}")
        print("\nSummary statistics (first 5 columns):")
        print(self.features.iloc[:, :5].describe())

        # Loop through each column, with a conditional check inside the
        # loop — demonstrates "loop + conditional statement within loop".
        missing_columns = []
        for column in self.features.columns:
            missing_count = self.features[column].isnull().sum()
            if missing_count > 0:
                missing_columns.append(column)
                print(f"  Warning: '{column}' has {missing_count} missing values.")

        if not missing_columns:
            print("No missing values found in any column.")

        RESULTS_SUMMARY["missing_values_found"] = len(missing_columns)

        # Class balance check
        print("\nClass distribution:")
        for class_value, count in self.target.value_counts().items():
            label = self.target_names[class_value]
            print(f"  {label}: {count} samples")

    # ------------------------------------------------------------------
    # DATA CLEANING & PREPROCESSING
    # ------------------------------------------------------------------
    def clean_data(self):
        """
        Clean the dataset: drop duplicate rows and fill/drop missing
        values if any are found. Uses conditional logic to decide what
        cleaning action is needed.
        """
        print("\n--- Data Cleaning ---")
        duplicate_count = self.features.duplicated().sum()
        RESULTS_SUMMARY["duplicate_rows_found"] = int(duplicate_count)

        if duplicate_count > 0:
            self.features = self.features.drop_duplicates()
            self.target = self.target.loc[self.features.index]
            print(f"Removed {duplicate_count} duplicate rows.")
        else:
            print("No duplicate rows found.")

        # Loop through columns again with a conditional: fill any
        # missing numeric values with the column mean.
        for column in self.features.columns:
            if self.features[column].isnull().any():
                mean_value = self.features[column].mean()
                self.features[column] = self.features[column].fillna(mean_value)
                print(f"Filled missing values in '{column}' with mean ({mean_value:.4f}).")

        print("Data cleaning complete.")

    def preprocess(self):
        """
        Standardize features (mean = 0, std = 1). This is required
        before PCA because PCA is sensitive to the scale of variables.
        """
        try:
            scaler = StandardScaler()
            self.scaled_features = scaler.fit_transform(self.features)
            print("\nFeatures standardized successfully.")
        except Exception as error:
            raise RuntimeError(f"Failed during preprocessing: {error}") from error
        return self.scaled_features

    # ------------------------------------------------------------------
    # DATA MANIPULATION
    # ------------------------------------------------------------------
    def manipulate_data(self):
        """
        Demonstrate data manipulation: add a derived column and
        filter the dataset based on a condition, purely to show
        manipulation techniques used elsewhere in the workflow.
        """
        derived = self.features.copy()
        derived["mean_area_to_perimeter_ratio"] = (
            derived["mean area"] / derived["mean perimeter"]
        )
        high_ratio_count = (derived["mean_area_to_perimeter_ratio"] > 5).sum()
        print(
            f"\nData manipulation example: "
            f"{high_ratio_count} samples have a mean area-to-perimeter "
            f"ratio greater than 5."
        )
        return derived

    # ------------------------------------------------------------------
    # PCA (DIMENSIONALITY REDUCTION)
    # ------------------------------------------------------------------
    def apply_pca(self):
        """Fit PCA on the scaled features and reduce to n_components."""
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.principal_components = self.pca.fit_transform(self.scaled_features)
        return self.principal_components

    def report_variance(self):
        """
        Report explained variance per component. Loop with a
        conditional statement flags components with low variance.
        """
        print("\n--- Explained Variance Ratio ---")
        for i, ratio in enumerate(self.pca.explained_variance_ratio_, start=1):
            component_name = f"PC{i}"
            RESULTS_SUMMARY["explained_variance"][component_name] = round(float(ratio), 4)

            if ratio < VARIANCE_ALERT_THRESHOLD:
                print(f"{component_name}: {ratio:.4f} (Note: low contribution)")
            else:
                print(f"{component_name}: {ratio:.4f} ({ratio * 100:.2f}%)")

        total_variance = sum(self.pca.explained_variance_ratio_)
        print(f"Total variance captured: {total_variance * 100:.2f}%")

    def show_top_contributing_features(self, top_n=5):
        """Print the features that contribute most to each component."""
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i + 1}" for i in range(self.pca.n_components_)],
            index=self.features.columns,
        )
        print("\n--- Top Contributing Features per Component ---")
        for col in loadings.columns:
            top_features = loadings[col].abs().sort_values(ascending=False).head(top_n)
            print(f"\n{col}:")
            for feature, weight in top_features.items():
                print(f"  {feature}: {loadings.loc[feature, col]:.4f}")

    # ------------------------------------------------------------------
    # EXPORT TO CSV
    # ------------------------------------------------------------------
    def export_to_csv(self, filename="pca_reduced_data.csv"):
        """
        Export the PCA-reduced dataset to a CSV file.
        Wrapped in try/except to handle potential file write errors.
        """
        try:
            pca_df = pd.DataFrame(
                self.principal_components,
                columns=[f"PC{i + 1}" for i in range(self.n_components)],
            )
            pca_df["diagnosis"] = self.target.reset_index(drop=True).map(
                dict(enumerate(self.target_names))
            )
            pca_df.to_csv(filename, index=False)
            print(f"\nPCA-reduced dataset saved as '{filename}'")
        except (IOError, OSError) as error:
            print(f"Error: Could not export CSV file. Details: {error}")

    # ------------------------------------------------------------------
    # DATA VISUALIZATION
    # ------------------------------------------------------------------
    def plot_pca_result(self, filename="pca_scatter_plot.png"):
        """Scatter plot of the 2 principal components colored by diagnosis."""
        plt.figure(figsize=(8, 6))
        colors = ["#d62728", "#2ca02c"]  # red = malignant, green = benign

        for class_value, color, label in zip([0, 1], colors, self.target_names):
            mask = self.target.reset_index(drop=True) == class_value
            plt.scatter(
                self.principal_components[mask, 0],
                self.principal_components[mask, 1],
                c=color,
                label=label,
                alpha=0.7,
                edgecolor="k",
                s=40,
            )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA of Breast Cancer Dataset (2 Components)")
        plt.legend()
        plt.tight_layout()

        try:
            plt.savefig(filename, dpi=150)
            print(f"Scatter plot saved as '{filename}'")
        except (IOError, OSError) as error:
            print(f"Error: Could not save plot. Details: {error}")
        finally:
            plt.close()

    # ------------------------------------------------------------------
    # BONUS: LOGISTIC REGRESSION
    # ------------------------------------------------------------------
    def run_logistic_regression(self):
        """
        Bonus: Train a Logistic Regression model on the 2 PCA
        components to predict cancer diagnosis, and report metrics.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.principal_components,
                self.target,
                test_size=TEST_SIZE,
                random_state=self.random_state,
                stratify=self.target,
            )

            self.model = LogisticRegression(random_state=self.random_state)
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            RESULTS_SUMMARY["logistic_regression_accuracy"] = round(float(accuracy), 4)

            print("\n--- Logistic Regression Results (Bonus) ---")
            print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, predictions))
            print("\nClassification Report:")
            print(classification_report(y_test, predictions))

        except ValueError as error:
            # Exception handling around model training/prediction
            print(f"Error while training logistic regression model: {error}")

        return self.model


def main():
    """Run the full PCA workflow end-to-end."""
    analyzer = CancerPCAAnalyzer(n_components=N_COMPONENTS, random_state=RANDOM_STATE)

    # Data preparation
    analyzer.load_data()

    # Data exploration
    analyzer.explore_data()

    # Data cleaning and preprocessing
    analyzer.clean_data()
    analyzer.preprocess()

    # Data manipulation (demonstration)
    analyzer.manipulate_data()

    # PCA dimensionality reduction
    analyzer.apply_pca()
    analyzer.report_variance()
    analyzer.show_top_contributing_features()

    # Export results
    analyzer.export_to_csv()

    # Visualization
    analyzer.plot_pca_result()

    # Bonus: prediction
    analyzer.run_logistic_regression()

    # Print final summary dictionary
    print("\n--- Run Summary (dictionary) ---")
    for key, value in RESULTS_SUMMARY.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
