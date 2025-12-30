# 🎓 Predictive Core: Academic Performance Insight Engine

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-v1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="gctu_logo.png" width="300" alt="Predictive Core Logo">
</p>

## 🌟 Overview

**Predictive Core** is a high-fidelity predictive analytics platform designed to anticipate student academic outcomes. Built for the Ghana Communication Technology University (GCTU), this engine transforms behavioral markers—such as study habits, attendance, and historical performance—into precise, actionable intelligence.

---

## 🧠 Intelligence Architecture

### The Algorithm: Random Forest Ensemble
Predictive Core is powered by a **Random Forest (RF) Classifier**, one of the most robust and accurate algorithms in modern Machine Learning for tabular data.

#### How it Works:
1.  **Ensemble Learning**: Unlike a single decision tree which can be biased, Random Forest builds **100 individual decision trees** (estimators).
2.  **Bagging (Bootstrap Aggregating)**: Each tree is trained on a random subset of the data and a random selection of features. This ensures that no single feature dominates the model and reduces "overfitting."
3.  **Consensus Voting**: When a prediction is requested, all 100 trees "vote" on the outcome. The final result is the majority consensus.
4.  **Local Factor Analysis**: Predictive Core uses the model's internal **Feature Importance** (computed via Gini Impurity) to explain *why* a specific student is at risk, providing transparent and ethical AI results.

---

## 🖥️ Local Development Guide (VS Code)

Follow these steps to set up and run Predictive Core on your local machine using **Visual Studio Code**.

### 1. Prerequisites
*   **VS Code**: [Download here](https://code.visualstudio.com/)
*   **Python 3.10+**: [Download here](https://www.python.org/)
*   **Git**: [Download here](https://git-scm.com/)

### 2. Setup Procedure
1.  **Clone the Repository**:
    Open the VS Code Terminal (`Ctrl + ``) and run:
    ```bash
    git clone https://github.com/your-username/predictive-core.git
    cd predictive-core
    ```
2.  **Create a Virtual Environment**:
    This keeps your project dependencies isolated.
    ```bash
    python -m venv venv
    ```
3.  **Activate Environment**:
    *   **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
    *   **Mac/Linux**: `source venv/bin/activate`
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Running the Application
To launch the dashboard, run the following command in the VS Code terminal:
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 🚀 Key Features

- **🔮 Neural Analyzer**: Individual performance simulations with probability confidence and local factor analysis.
- **📂 Cohort Processor**: Batch analysis of entire class datasets with automated data cleaning and success distribution charts.
- **📊 Technical Specs**: Real-time display of model metrics, including Confusion Matrices and System Latency audits.
- **💎 Elite UI**: Premium glassmorphism design with academic credentials and interactive Plotly visualizations.

---

## 📡 Predictive Vectors (Model Inputs)

| Vector | Description |
| :--- | :--- |
| **Study Habits** | Average hours of focused daily study (1-5 hrs). |
| **Academic History** | Count of previous failed course attempts (0-4). |
| **Attendance** | Total absences recorded during the academic cycle. |
| **G1 & G2 Grades** | Performance scores from the first and second assessment periods. |

---

<p align="center">
  Made with ❤️ by the <b>GCTU AI Research Team</b>
</p>