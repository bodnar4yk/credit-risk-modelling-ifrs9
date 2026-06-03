import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# =====================================================================
# 1. DATA LOADING & IMMUTABLE CLEANING (CRITICAL FIX FOR NaNs)
# =====================================================================
# Load the original benchmark dataset
df_raw = pd.read_csv('cs-training.csv')

# Select only historical default cases
df_lkp = df_raw[df_raw['SeriousDlqin2yrs'] == 1].copy()

if len(df_lkp) == 0:
    raise ValueError("No default cases found. Ensure cs-training.csv is in the workspace root.")

# --- CRITICAL FIX: Clean raw dependencies before generating LGD/RR targets ---
# Fill raw columns with median values to eliminate NaN propagation into the target vectors
df_lkp['NumberRealEstateLoansOrLines'] = df_lkp['NumberRealEstateLoansOrLines'].fillna(df_lkp['NumberRealEstateLoansOrLines'].median())
df_lkp['NumberOfDependents'] = df_lkp['NumberOfDependents'].fillna(df_lkp['NumberOfDependents'].median())
df_lkp['NumberOfTime30-59DaysPastDueNotWorse'] = df_lkp['NumberOfTime30-59DaysPastDueNotWorse'].fillna(df_lkp['NumberOfTime30-59DaysPastDueNotWorse'].median())

# --- Realism Simulation: Generating Recovery Rate (RR) ---
np.random.seed(42)
n_samples = len(df_lkp)

# Base probability of any recovery occurring
prob_recovery = 0.3 + 0.4 * (df_lkp['NumberRealEstateLoansOrLines'] > 0).astype(int) \
                    - 0.1 * df_lkp['NumberOfTime30-59DaysPastDueNotWorse'].clip(0, 3) / 3
prob_recovery = np.clip(prob_recovery, 0.05, 0.95)

# Stage 1 Target: Binary indicator (1 if some money recovered, 0 if total loss)
df_lkp['Recovery_Observed'] = np.random.binomial(1, prob_recovery)

# Stage 2 Target: Continuous Recovery Rate (0% to 100%) given that recovery occurred
raw_rr = 0.4 + 0.3 * (df_lkp['NumberRealEstateLoansOrLines'] > 0).astype(int) \
             - 0.05 * df_lkp['NumberOfDependents'].clip(0, 4) + np.random.normal(0, 0.15, n_samples)
df_lkp['Recovery_Rate'] = np.where(df_lkp['Recovery_Observed'] == 1, np.clip(raw_rr, 0.01, 0.99), 0.0)

# Final LGD Target: LGD = 1 - Recovery Rate
df_lkp['LGD'] = 1.0 - df_lkp['Recovery_Rate']

# =====================================================================
# 2. FEATURE ENGINEERING & DATA SPLIT
# =====================================================================
features = [
    'RevolvingUtilizationOfUnsecuredLines', 
    'age', 
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfOpenCreditLinesAndLoans', 
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents'
]

# Ensure X feature matrix is also safe and clean
X = df_lkp[features].copy()
for col in features:
    X[col] = X[col].fillna(X[col].median())

y_binary = df_lkp['Recovery_Observed']
y_continuous = df_lkp['Recovery_Rate']

# Split data into Train and Test sets (80/20)
X_train, X_test, y_train_bin, y_test_bin, y_train_cont, y_test_cont = train_test_split(
    X, y_binary, y_continuous, test_size=0.2, random_state=42
)

print(f"📊 LGD Modeling Dataset Initialized. Default Observations: {n_samples}")
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

# =====================================================================
# 3. STAGE 1 MODEL: PREDICTING THE PROBABILITY OF RECOVERY (Logistic)
# =====================================================================
model_stage1 = LogisticRegression(random_state=42, max_iter=1000)
model_stage1.fit(X_train, y_train_bin)

stage1_preds_train = model_stage1.predict_proba(X_train)[:, 1]
stage1_preds_test = model_stage1.predict_proba(X_test)[:, 1]
print("--- STAGE 1 PERFORMANCE (Binary Classification) ---")
print(f"Train AUC-ROC : {roc_auc_score(y_train_bin, stage1_preds_train):.4f}")
print(f"Test AUC-ROC  : {roc_auc_score(y_test_bin, stage1_preds_test):.4f}\n")

# =====================================================================
# 4. STAGE 2 MODEL: ESTIMATING THE RECOVERY RATE (Linear Regression)
# =====================================================================
# Filter training data to include ONLY observations where recovery actually occurred
train_recovery_mask = (y_train_bin == 1)
X_train_stage2 = X_train[train_recovery_mask]
y_train_stage2 = y_train_cont[train_recovery_mask]

model_stage2 = LinearRegression()
model_stage2.fit(X_train_stage2, y_train_stage2)

# =====================================================================
# 5. TWO-STAGE COMBINED FORECASTING (INFERENCE)
# =====================================================================
def predict_lgd(X_matrix):
    # P(Recovery > 0)
    p_recovery = model_stage1.predict_proba(X_matrix)[:, 1]
    
    # E(Recovery Rate | Recovery > 0)
    expected_rr_amount = model_stage2.predict(X_matrix)
    expected_rr_amount = np.clip(expected_rr_amount, 0.0, 1.0) # Economic boundaries
    
    # Final Unconditional Expected Recovery Rate = P(Recovery) * E(RR | Recovery)
    final_expected_rr = p_recovery * expected_rr_amount
    
    # LGD = 1 - Final Expected Recovery Rate
    final_lgd = 1.0 - final_expected_rr
    return final_lgd

# Compute final LGD outputs
final_lgd_train = predict_lgd(X_train)
final_lgd_test = predict_lgd(X_test)

# Calculate actual true LGD for evaluation
true_lgd_train = 1.0 - y_train_cont
true_lgd_test = 1.0 - y_test_cont

# =====================================================================
# 6. FINAL LGD VALIDATION METRICS
# =====================================================================
print("--- FINAL TWO-STAGE LGD MODEL PERFORMANCE ---")
print(f"Train RMSE (Root Mean Squared Error): {np.sqrt(mean_squared_error(true_lgd_train, final_lgd_train)):.4f}")
print(f"Test RMSE  (Root Mean Squared Error): {np.sqrt(mean_squared_error(true_lgd_test, final_lgd_test)):.4f}")
print(f"Test R2 Score (Explaining Variance) : {r2_score(true_lgd_test, final_lgd_test):.4f}")

# Display statistical profile of predicted LGD
print("\n--- PREDICTED LGD DISTRIBUTION PROFILE (TEST) ---")
print(pd.Series(final_lgd_test).describe())
