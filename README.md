# Applied Machine Learning & Predictive Modeling Benchmarks

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Mlxtend](https://img.shields.io/badge/Mlxtend-Market_Basket-2E8B57?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

## Executive Summary

This repository centralizes foundational classical Machine Learning benchmarks and engineering experiments spanning **supervised classification**, **unsupervised clustering**, and **frequent pattern mining**. Designed with a focus on algorithmic interpretability and end-to-end data pipeline execution, these implementations demonstrate practical proficiency across data ingestion, missing-value imputation, categorical and transactional encoding, feature scaling, model training, holdout validation, and schema-aligned inference. Together, the experiments establish a rigorous baseline for predictive modeling, customer segmentation, and basket recommendation systems.

---

## Implemented Models & Datasets

The table below summarizes the experimental benchmarks, algorithmic configurations, datasets, and exact evaluation metrics obtained across the notebooks in this repository:

| Project / Notebook | Primary Algorithm(s) | Dataset | Key Metrics & Experimental Output |
| :--- | :--- | :--- | :--- |
| [`01_supervised_classification_benchmarks.ipynb`](./01_supervised_classification_benchmarks.ipynb) *(Exp. 1)* | **K-Nearest Neighbors** (`KNeighborsClassifier`, `k=5`) | Iris Botanical Dataset (`iris.data`) | **Accuracy: 100.0%** (`80/20` holdout split, `random_state=42`) |
| [`01_supervised_classification_benchmarks.ipynb`](./01_supervised_classification_benchmarks.ipynb) *(Exp. 2)* | **Decision Tree Classifier** (`DecisionTreeClassifier`, `max_depth=3`) | Iris Botanical Dataset (`iris.data`) | **Accuracy: 100.0%** (`75/25` holdout split, tree visualization & CLI inference) |
| [`01_supervised_classification_benchmarks.ipynb`](./01_supervised_classification_benchmarks.ipynb) *(Exp. 3)* | **Gaussian Naive Bayes** (`GaussianNB`) | Titanic Survival (`seaborn`) | **Accuracy: 78.48%** (`75/25` holdout split, `predict_proba` calibrated output) |
| [`02_decision_tree_rule_induction.ipynb`](./02_decision_tree_rule_induction.ipynb) | **Decision Tree Classifier** (`DecisionTreeClassifier`, `max_depth=2`) | Weather / Play Tennis (`clima_tenis.csv`) | **Rule Tree Fitting & Online Inference** (Full-dataset decision boundary visualization) |
| [`03_knn_distance_classification.ipynb`](./03_knn_distance_classification.ipynb) | **K-Nearest Neighbors** (`KNeighborsClassifier`, `k=3`) | Weather / Play Tennis (`clima_tenis.csv`) | **Accuracy: 60.0%** (`70/30` holdout split, `random_state=42`) |
| [`04_gaussian_naive_bayes_baseline.ipynb`](./04_gaussian_naive_bayes_baseline.ipynb) | **Gaussian Naive Bayes** (`GaussianNB`) | Weather / Play Tennis (`clima_tenis.csv`) | **Accuracy: 80.0%** (`70/30` holdout split, `random_state=42`) |
| [`05_credit_risk_clustering_analysis.ipynb`](./05_credit_risk_clustering_analysis.ipynb) | **K-Means** (`k=3`) & **Agglomerative Clustering** (Ward Linkage) | Credit Risk Profiles (`datasets/credito.csv`) | **3 Customer Segments** (Euclidean Ward dendrogram & standardized 2D income/age projection) |
| [`06_retail_market_basket_apriori.ipynb`](./06_retail_market_basket_apriori.ipynb) | **Apriori** & **Association Rules** (`mlxtend`) | Supermarket Retail (`datasets/produtos.csv`) | **Support $\ge$ 0.50 \| Confidence: 1.00 \| Peak Lift: 2.00** |
| [`07_unsupervised_clustering_and_association_rules.ipynb`](./07_unsupervised_clustering_and_association_rules.ipynb) *(Exp. 1 & 2)* | **K-Means** (`k=3`) & **Agglomerative Clustering** (Ward Linkage) | Titanic Demographics (`seaborn`) | **3 Socioeconomic Tiers** (Cluster survival rates: **30.18%**, **53.36%**, and **71.43%**) |
| [`07_unsupervised_clustering_and_association_rules.ipynb`](./07_unsupervised_clustering_and_association_rules.ipynb) *(Exp. 3)* | **K-Means** (`k=3`) vs. **Agglomerative Clustering** (Ward Linkage) | Iris Dataset (`sklearn.datasets`) | **Cross-Method Agreement: 98.67%** (`148/150` samples aligned in contingency crosstab) |
| [`07_unsupervised_clustering_and_association_rules.ipynb`](./07_unsupervised_clustering_and_association_rules.ipynb) *(Exp. 4)* | **Apriori** & **Association Rules** (`mlxtend`) | Retail Market Basket (In-Memory) | **Support: 0.60 \| Confidence: 0.75–1.00 \| Peak Lift: 1.25** |
| [`07_unsupervised_clustering_and_association_rules.ipynb`](./07_unsupervised_clustering_and_association_rules.ipynb) *(Exp. 5)* | **Apriori** & **Association Rules** (`mlxtend`) | E-Commerce Electronics Cross-Sell | **Support $\ge$ 0.20 \| Confidence: 1.00 \| Lift: 1.50** (Multi-metric filtering & ranking) |

---

## Core Competencies Demonstrated

* **Feature Scaling to Prevent Distance-Based Bias:** Application of `StandardScaler` (`z`-score normalization) on continuous variables with disparate magnitudes—such as `age` vs. `fare` in the Titanic dataset and `idade` vs. `renda` in the Credit dataset—prior to running Euclidean-distance algorithms (`KMeans` and `AgglomerativeClustering`).
* **Missing Data Imputation & Feature Pruning:** Principled handling of incomplete records by pruning high-null columns (`deck`) and applying mean imputation (`fillna(df['age'].mean())`) on continuous demographic features before model fitting.
* **Categorical & Transactional Encoding:** Transformation of nominal and ordinal variables into numerical feature spaces using Pandas `pd.get_dummies()` (including `drop_first=True` to mitigate dummy-variable multicollinearity in Naive Bayes) and `mlxtend.preprocessing.TransactionEncoder` for sparse boolean basket matrices.
* **Schema-Aligned Inference & Probabilistic Scoring:** Implementation of inference functions that dynamically encode single-record payloads and enforce training-time feature alignment via `.reindex(columns=X_train.columns, fill_value=0)`, paired with `predict_proba()` to expose calibrated class probabilities rather than raw hard labels alone.
* **Model Interpretability & Tree Pruning:** Constraining decision tree complexity via `max_depth` (`max_depth=2` and `max_depth=3`) to prevent overfitting on small/medium tabular datasets, coupled with visual inspection of split criteria and leaf purity using `sklearn.tree.plot_tree`.
* **Unsupervised Partitioning & Cross-Algorithm Validation:** Comparative evaluation of centroid-based (`KMeans` with explicit `n_init=10`) and hierarchical (`AgglomerativeClustering` with `ward` variance-minimizing linkage) clustering, audited via SciPy truncated dendrograms (`truncate_mode='lastp'`) and `pd.crosstab` contingency matrices.
* **Market Basket Analysis & Cross-Selling Optimization:** Extraction of frequent itemsets via the `apriori` algorithm and multi-criteria rule pruning across **Support**, **Confidence**, and **Lift** thresholds (`confidence >= 0.8`, `lift > 1.2` / `1.5`) to surface high-affinity product bundles.

---

## Repository Structure

Below is the repository layout alongside the recommended modular structure for scaling into categorized benchmark suites:

```text
classical-ml-supervised-benchmarks/
├── README.md
├── requirements.txt
│
├── 01_supervised_classification_benchmarks.ipynb       # Supervised suite: KNN, Decision Tree (Iris) & GaussianNB (Titanic)
├── 02_decision_tree_rule_induction.ipynb               # Decision Tree rule induction & visualization (Play Tennis)
├── 03_knn_distance_classification.ipynb                # K-Nearest Neighbors classification benchmark (Play Tennis)
├── 04_gaussian_naive_bayes_baseline.ipynb              # Gaussian Naive Bayes baseline (Play Tennis)
├── 05_credit_risk_clustering_analysis.ipynb            # K-Means & Ward Hierarchical Clustering (Credit Risk)
├── 06_retail_market_basket_apriori.ipynb               # Apriori frequent itemsets & association rules (Supermarket)
├── 07_unsupervised_clustering_and_association_rules.ipynb # Unsupervised suite: Clustering (Titanic/Iris) & Apriori (Retail/E-Commerce)
│
└── datasets/                                           # Local tabular data sources (recommended consolidation)
    ├── iris.data
    ├── clima_tenis.csv
    ├── credito.csv
    └── produtos.csv
```

> **Suggested Organization for Production/Portfolio Scaling:**
> ```text
> classical-ml-supervised-benchmarks/
> ├── data/
> │   ├── raw/                                            # iris.data, clima_tenis.csv, credito.csv, produtos.csv
> │   └── processed/
> ├── notebooks/
> │   ├── 01_supervised_classification/
> │   │   ├── 01_supervised_classification_benchmarks.ipynb
> │   │   ├── 02_decision_tree_rule_induction.ipynb
> │   │   ├── 03_knn_distance_classification.ipynb
> │   │   └── 04_gaussian_naive_bayes_baseline.ipynb
> │   └── 02_unsupervised_and_pattern_mining/
> │       ├── 05_credit_risk_clustering_analysis.ipynb
> │       ├── 06_retail_market_basket_apriori.ipynb
> │       └── 07_unsupervised_clustering_and_association_rules.ipynb
> ├── README.md
> └── requirements.txt
> ```

---

## How to Run (Local Setup)

### 1. Clone the repository and create a virtual environment

```bash
git clone <repository-url>
cd classical-ml-supervised-benchmarks

python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch Jupyter Lab / Notebook

```bash
jupyter lab
# or
jupyter notebook
```
