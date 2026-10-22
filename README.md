# Iris Metric inspector

> Automated botanical classification engine leveraging nearest-neighbor metric analysis and continuous terminal evaluation loops.

---

## Overview / Value Proposition

Testing machine learning models during early exploration often requires repetitive script execution, manual dataset loading, and cumbersome parameter tweaking. This creates friction when validating baseline algorithms or inspecting vector spaces. `iris-metric-inspector` eliminates this operational overhead by combining automated dataset partitioning, model training, accuracy evaluation, and an interactive prediction loop into a single, cohesive command-line utility.

---

## Under the Hood / How It Works

The execution pipeline operates through sequential, well-defined phases managed within standard Python streams:

1. **Dataset Ingestion & Partitioning:** The engine loads benchmark flower measurements using `scikit-learn`'s `load_iris` utility. Features and target labels are separated into matrix arrays and partitioned via `train_test_split` with a fixed pseudo-random seed (`random_state=42`) to guarantee reproducible evaluation splits.
2. **Estimator Training:** A `KNeighborsClassifier` instance is initialized with three neighbors ($k=3$) and fitted against the training subset, computing distance metrics across multidimensional feature spaces.
3. **Statistical Validation:** The model evaluates test data, generating exact accuracy scores and detailed classification reports covering precision, recall, and F1-metrics per target class.
4. **Interactive Prediction Loop:** The program enters an infinite polling loop on standard input, parsing comma- or space-separated floating-point measurements through list comprehensions to deliver instant species classifications.

---

## Key Features

* **Reproducible Splits:** Deterministic training and testing partitions ensuring consistent validation metrics across runs.
* **Granular Evaluation:** Comprehensive accuracy scoring and classification report generation using standard metric utilities.
* **Fault-Tolerant Input Parsing:** Graceful handling of malformed user inputs, dimensional mismatches, and type coercion errors via targeted exception blocks.
* **Zero-Configuration Startup:** Executes entirely out-of-the-box using standard library constructs and minimal scientific computing dependencies.

---

## Tech Stack & Core Dependencies Breakdown

* **Python 3.10+:** Target runtime environment utilizing modern type hints and clean exception handling semantics.
* **`scikit-learn` (v1.0+):** Core machine learning library providing the Iris dataset loader, train-test splitting utilities, KNeighbors classifier, and evaluation metrics.
* **Python Standard Library (`sys`):** Utilized for error stream redirection and clean process termination codes.

---

## Environment & Web-Based Quick Start

### Running via GitHub Codespaces (Browser-Only)
1. Navigate to the repository main page on GitHub.
2. Click the green **Code** button and select the **Codespaces** tab.
3. Click **Create codespace on main** to spin up a fully configured cloud development environment.
4. Once the terminal loads, execute the application:
   ```bash
   python main.py
   ```

### Local Virtual Environment Setup
If running locally, set up an isolated environment using standard tools:

```bash
# Clone the repository
git clone [https://github.com/username/iris-metric-inspector.git](https://github.com/username/iris-metric-inspector.git)
cd iris-metric-inspector

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install scikit-learn

# Run the utility
python main.py
```

## Repository Structure

```bash
iris-metric-inspector/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
└── main.py
```

## Roadmap

**[ ] Model Serialization:** Integrate joblib persistence to save trained estimators to disk, avoiding retraining overhead on subsequent executions.

**[ ] Type Stubs & Strict Typing:** Apply comprehensive type annotations across all functions and integrate mypy static type checking into the CI pipeline.

**[ ] Configuration Extensibility:** Introduce a lightweight YAML configuration parser to allow dynamic tuning of hyper-parameters (e.g., neighbor count, distance metrics) without modifying source code.
