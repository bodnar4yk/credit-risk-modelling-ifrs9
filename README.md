# Credit Risk Modelling Architecture (IFRS 9 / Basel IRB)

This repository contains a comprehensive framework for credit risk assessment, focusing on the development and validation of **PD (Probability of Default)**, **LGD (Loss Given Default)**, and **EAD (Exposure at Default)** internal models for retail banking portfolios.

The repository is structured logically to separate the different stages of the credit risk pipeline, aligning with standard industry validation techniques.

---

## 📁 Repository Structure
* `1_PD_Model/` — Application/Behavioral Scorecard development using Logistic Regression on Weight of Evidence (WoE) transformed variables.
* `2_LGD_Model/` — Recovery rate and Loss Given Default estimation.
* `3_EAD_Model/` — *(In Progress)* Credit Conversion Factor (CCF) and Exposure at Default forecasting.

---

## ⚙️ Data Sourcing & Kaggle API Integration

To prevent storing massive raw credit data files within the Git history, the data pipeline is integrated directly with the Kaggle API. The dataset utilized is the industrial benchmark **"Give Me Some Credit"**.

### Prerequisites & API Token Configuration:
To establish a secure connection, the Kaggle API token (`kaggle.json`) must be placed in the core system directory of your operating system. The script automatically looks for this token to authenticate the session:
* **Windows:** `C:\Users\<Your_Username>\.kaggle\kaggle.json`
* **Linux / macOS:** `~/.kaggle/kaggle.json`

Once the token is placed in your local system path, install the official Kaggle CLI via pip (`pip install kaggle`). The pipeline will then execute the programmatic session authentication and download the dataset seamlessly without exposing raw credentials:

```python
import kaggle

# Authenticate the session using the locally stored system API token
kaggle.api.authenticate() 

# Programmatically fetch and unzip the data files into the workspace root
kaggle.api.dataset_download_files(
    'brycecf/give-me-some-credit-dataset', 
    path='.', 
    unzip=True
)

```
## 💳 Component 1: Probability of Default (PD) Scorecard

The first completed component is a production-grade Credit Scorecard built using an interpretable, regulator-ready statistical framework.

### Methodology & Technical Design:

1. **Advanced Feature Cleaning:** Outliers are managed using statistical capping (99th percentile clipping for volatile income inputs). Minimum age limits are strictly enforced to align with retail lending policies.

2. **Optimal Dynamic Binning:** Replaced subjective arbitrary bucketing with mathematical split thresholds determined via shallow Decision Trees (enforcing a minimum 5% sample leaf node limit to ensure Basel stability constraints).

3. **Weight of Evidence (WoE) & Information Value (IV):** Variables are screened and prioritized based on their predictive strength. Weak or unstable features are filtered out automatically using the Information Value index.

4. **Parametric Points Scaling: Rather than outputting raw decimals, the log-odds coefficients from the Logistic Regression model are scaled into an intuitive, operational credit point system using the following parametric anchors:**

* **Base Score: 600 points**

* **Base Odds: 50:1**

* **Points to Double the Odds (PDO): 20 points**

5. **Model Interpretability:** Uses an exact monotonic linear mapping via Logistic Regression, ensuring absolute transparency for corporate risk committees and national regulators.

### Model Performance Metrics:

The model demonstrates high discriminative power and outstanding stability between training and out-of-sample test splits:

* **Train AUC-ROC: ~0.86+ | Train GINI: ~0.72+**

* **Test AUC-ROC: ~0.85+  | Test GINI: ~0.70+**
(Note: These figures show zero signs of overfitting, securing reliable risk forecasting on future applicant vintages).

## 📉 Component 2: Loss Given Default (LGD) Estimation 
LGD modeling targets the conditional loss percentage incurred by the financial institution once an asset enters a default state. Due to the classical bi-modal configuration of recovery realization (frequent absolute recoveries or absolute losses), an advanced Two-Stage Modeling Architecture was engineered.

### Mathematical Framework & Inference Flow:
1. **Stage 1 (Probability of Recovery):** A Logistic Regression classifier estimates the unconditional probability that any recovery actions will yield cash returns ($P(Recovery > 0)$).
2. **Stage 2 (Conditional Recovery Severity):** A linear regression model calculates the conditional numeric intensity of recovery rates ($E(RR | Recovery > 0)$) restricted exclusively to accounts that cleared Stage 1.
3. **Unconditional Integration:** The finalized Loss Given Default prediction is consolidated mathematically utilizing joint expectations:

$$\text{Expected RR} = P(\text{Recovery} > 0) \times E(\text{RR} \mid \text{Recovery} > 0)$$

$$\text{LGD} = 1.0 - \text{Expected RR}$$

### Validation Architecture:

The model constraints predicted values strictly inside coherent economic boundaries ($LGD \in [0.0, 1.0]$) and validates accuracy utilizing RMSE (Root Mean Squared Error) to evaluate structural prediction errors against simulated historical recovery workouts.

### 💼 Business Application & Credit Strategy

The scaled points system and LGD profiles derived from these models can be operationalized into a retail credit workflow:

* **Risk Segmentation:** Customers with scores above 640 are classified as Low Risk (eligible for auto-approval and premium pricing limits), while applicants below 550 represent High Default risk and are routed for immediate rejection or manual underwriting.
  
* **Expected Loss (EL) Calibration:** The PD and LGD metrics serve as direct upstream components for the comprehensive IFRS 9 Expected Loss framework ($EL = PD \times LGD \times EAD$).

## 🛠️ Technological Stack & Dependencies

* **Core Environment: Python 3.x**

* **Data Processing & Analytics: Pandas, NumPy**

* **Machine Learning & Modeling: Scikit-learn (Decision Trees, Logistic Regression)**

* **Data Sourcing: Kaggle API**

To replicate the environment and run the files, install the required libraries:
    ```
    pip install -r requirements.txt
    ```
  
## 🚀 Quick Start: How to Run

1. **Clone the repository:**
    ```bash
       git clone https://github.com/bodnar4yk/credit-risk-modelling-ifrs9.git
       cd credit-risk-modelling-ifrs9
    ```
2. **Setup your Kaggle Token:**
    Place your kaggle.json file inside your core system path (~/.kaggle/ or C:\Users\<User>\.kaggle\).

4. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
5. **Execute the PD Model:**
   ```
   python 1_PD_Model/credit_scorecard.py
   ```
6. **Execute the LGD Model:**
   ```
   python 2_LGD_Model/lgd_model.py
   ```
