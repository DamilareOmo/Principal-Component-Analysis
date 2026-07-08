# Principal Component Analysis
### Anderson Cancer Center — Dimensionality Reduction Project

## Project Overview
The Anderson Cancer Center is facing a growing number of referrals and needs
to identify the essential variables driving cancer diagnoses in order to
support a model used for securing donor funding. This project uses
**Principal Component Analysis (PCA)** on the Breast Cancer Wisconsin
dataset (available in `sklearn.datasets`) to reduce 30 clinical features
down to **2 principal components** that still capture most of the
variation in the data.

As a bonus, a **Logistic Regression** model is trained on the 2 PCA
components to demonstrate that diagnosis (malignant vs. benign) can still
be predicted accurately even after dimensionality reduction.

## Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569
- **Features:** 30 numeric features (e.g., mean radius, mean texture,
  mean concavity, worst perimeter, etc.)
- **Target:** Diagnosis — `0 = malignant`, `1 = benign`

## Programming Concepts Used (Modules 1–5)
This project was built to explicitly demonstrate the core concepts covered
in Modules 1 through 5:

| Concept | Where it appears in `pca_analysis.py` |
|---|---|
| **Variables** | Module-level constants (`RANDOM_STATE`, `N_COMPONENTS`, etc.) and instance variables inside the class |
| **Loops** | `for` loops in `explore_data()`, `clean_data()`, `report_variance()`, `show_top_contributing_features()`, `plot_pca_result()` |
| **Conditional statements within loops** | e.g. checking `if missing_count > 0` while looping through columns in `explore_data()`; `if ratio < VARIANCE_ALERT_THRESHOLD` while looping through explained variance in `report_variance()` |
| **Exception/error handling** | `try/except` blocks in `load_data()`, `preprocess()`, `export_to_csv()`, `plot_pca_result()`, and `run_logistic_regression()` |
| **Functions** | Every step of the workflow (loading, cleaning, exploring, PCA, exporting, plotting, modeling) is its own method/function |
| **Dictionary** | `RESULTS_SUMMARY` dictionary tracks run metadata and results (sample count, missing values, explained variance per component, model accuracy) and is printed at the end of the run |
| **Export to CSV** | `export_to_csv()` saves the PCA-reduced dataset to `pca_reduced_data.csv` |
| **R script to unzip the folder** | `unzip_project.R` extracts the submitted zip file using R's `unzip()` function, with error handling |
| **Class creation** | `CancerPCAAnalyzer` class encapsulates the entire workflow (object-oriented design) |
| **Data preparation** | `load_data()` loads and structures the raw dataset into features/target |
| **Data cleaning** | `clean_data()` checks for and removes duplicate rows, and fills any missing values with the column mean |
| **Data exploration** | `explore_data()` prints dataset shape, summary statistics, missing-value checks, and class distribution |
| **Data visualization** | `plot_pca_result()` generates a 2D scatter plot of the PCA components colored by diagnosis |
| **Data manipulation** | `manipulate_data()` creates a derived feature (area-to-perimeter ratio) and filters/counts based on a condition |
| **Data cleaning and preprocessing** | `preprocess()` standardizes all features with `StandardScaler` prior to PCA |

## What the Script Does (Workflow)
1. **Load data** — via the `CancerPCAAnalyzer.load_data()` method, wrapped in
   exception handling.
2. **Explore data** — shape, descriptive statistics, missing-value scan
   (loop + conditional), and class balance.
3. **Clean data** — removes duplicate rows and fills missing values (if any).
4. **Preprocess data** — standardizes features with `StandardScaler` (mean =
   0, std = 1), required because PCA is sensitive to feature scale.
5. **Manipulate data** — creates a derived column and filters on a condition
   as a demonstration of data manipulation.
6. **Apply PCA** — reduces the 30 standardized features to **2 principal
   components**.
7. **Report variance** — prints how much variance each component explains,
   flagging any component below a defined threshold.
8. **Show top contributing features** — identifies which original features
   most heavily influence each component.
9. **Export to CSV** — saves the reduced dataset (`PC1`, `PC2`, `diagnosis`)
   to `pca_reduced_data.csv`.
10. **Visualize** — saves a scatter plot (`pca_scatter_plot.png`) of PC1 vs.
    PC2, colored by diagnosis.
11. **(Bonus) Logistic Regression** — trains a classifier on the 2 PCA
    components and reports accuracy, a confusion matrix, and a
    classification report.
12. **Print summary dictionary** — displays the `RESULTS_SUMMARY` dictionary
    with key metrics from the run.

## Project Structure
```
pca_project/
├── pca_analysis.py         # Main Python script — run this file
├── PCA_Analysis.ipynb      # Jupyter Notebook version of the same workflow
├── unzip_project.R         # R script to unzip the submission folder
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── pca_scatter_plot.png    # Generated output (scatter plot of PC1 vs PC2)
└── pca_reduced_data.csv    # Generated output (reduced dataset)
```

## Two Ways to Run This Project
Both versions implement the exact same `CancerPCAAnalyzer` class and
workflow — use whichever fits your environment:

- **`pca_analysis.py`** — a standalone Python script, run from the
  command line.
- **`PCA_Analysis.ipynb`** — the same analysis broken into labeled Jupyter
  Notebook cells (data preparation, exploration, cleaning, preprocessing,
  manipulation, PCA, variance reporting, CSV export, visualization, and
  the bonus logistic regression), with outputs (tables, plots, and
  printed results) already rendered inline for easy review.

## How to Run

### 1. Unzip the submission (optional — using R)
If you received this project as a zip file, you can extract it using the
included R script:
```r
source("unzip_project.R")
```
This will extract all files into a folder called `pca_project_extracted`.

### 2. Install Python dependencies
Make sure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the script or notebook
**Option A — Python script:**
```bash
python pca_analysis.py
```

**Option B — Jupyter Notebook:**
```bash
jupyter notebook PCA_Analysis.ipynb
```
Then run all cells (`Cell → Run All` or `Restart & Run All`). The notebook
walks through data preparation, exploration, cleaning, preprocessing,
manipulation, PCA, variance reporting, CSV export, visualization, and the
bonus logistic regression step-by-step, with all outputs displayed inline.

### 4. Review the outputs
- **Console output:** data exploration summary, cleaning steps, explained
  variance, top contributing features, logistic regression metrics, and a
  final results dictionary.
- **`pca_scatter_plot.png`:** visualization of the dataset in 2D PCA space.
- **`pca_reduced_data.csv`:** the final reduced dataset (PC1, PC2,
  diagnosis).

## Results Summary
- The dataset had **no missing values or duplicate rows**, confirming it
  was already well-suited for analysis (validated during the cleaning step).
- **PC1** explains ~44% of the variance and is driven mainly by features
  related to concavity, compactness, and perimeter — tied to tumor shape
  irregularity.
- **PC2** explains ~19% of the variance, bringing the total to ~63% of
  variance captured with just 2 components (down from the original 30
  features).
- The scatter plot shows clear visual separation between malignant and
  benign tumors using just these 2 components.
- The bonus **Logistic Regression model** achieves approximately **95%
  accuracy** predicting diagnosis using only the 2 PCA components,
  showing that PCA preserved the signal needed for classification while
  drastically reducing dimensionality.

## Requirements
See `requirements.txt`:
- numpy
- pandas
- matplotlib
- scikit-learn

R (for `unzip_project.R`) requires only base R — no additional packages
needed.

## Author
Sodiq Omoniyi | Data Analyst | Milestone Assignment 2: Principal Component Analysis.
