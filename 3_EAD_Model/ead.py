import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# 1. DATA LOADING & SIMULATION OF CREDIT LINES (EAD/CCF TARGETS)
# =====================================================================
df_raw = pd.read_csv('cs-training.csv')

# EAD models are strictly calibrated on historical default events (SeriousDlqin2yrs == 1)
df_lkp = df_raw[df_raw['SeriousDlqin2yrs'] == 1].copy()

if len(df_lkp) == 0:
    raise ValueError("No default cases found. Ensure cs-training.csv is in the workspace root.")

# --- CRITICAL FIX: Clean raw data to prevent NaN propagation ---
df_lkp['MonthlyIncome'] = df_lkp['MonthlyIncome'].fillna(df_lkp['MonthlyIncome'].median())
df_lkp['NumberOfDependents'] = df_lkp['NumberOfDependents'].fillna(df_lkp['NumberOfDependents'].median())

# --- Realism Simulation: Generating Credit Limits & CCF ---
np.random.seed(42)
n_samples = len(df_lkp)

# Simulate Total Credit Limit based on Income and Open Lines
df_lkp['Credit_Limit'] = df_lkp['MonthlyIncome'].clip(1000, 20000) * np.random.uniform(1.5, 4.0, n_samples)

# Current Drawn Balance (Utilized amount at booking/observation date)
# Utilizing RevolvingUtilizationOfUnsecuredLines capped realistically
util_rate = df_lkp['RevolvingUtilizationOfUnsecuredLines'].clip(0.0, 1.0)
df_lkp['Current_Drawn'] = df_lkp['Credit_Limit'] * util_rate

# Unutilized Limit (The available cushion)
df_lkp['Unutilized_Limit'] = df_lkp['Credit_Limit'] - df_lkp['Current_Drawn']

# Simulate Actual CCF (Credit Conversion Factor) observed historically at default
# CCF depends on age, utilization behavior, and short-term delinquency shocks
base_ccf = 0.5 - 0.2 * (df_lkp['age'] / 100) + 0.3 * util_rate + np.random.normal(0, 0.2, n_samples)
df_lkp['True_CCF'] = np.clip(base_ccf, 0.0, 1.0)

# Calculate True EAD observed at the exact moment of default
# EAD = Drawn + CCF * Unutilized
df_lkp['True_EAD'] = df_lkp['Current_Drawn'] + (df_lkp['True_CCF'] * df_lkp['Unutilized_Limit'])

# =====================================================================
# 2. FEATURE ENGINEERING & DATA SPLIT
# =====================================================================
features = [
    'RevolvingUtilizationOfUnsecuredLines', 
    'age', 
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio', 
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans', 
    'NumberOfTime60-89DaysPastDueNotWorse'
]

X = df_lkp[features].copy()
for col in features:
    X[col] = X[col].fillna(X[col].median())

# Target vector is the historical CCF
y_ccf = df_lkp['True_CCF']

# Split into Train and Test sets
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y_ccf, df_lkp.index, test_size=0.2, random_state=42
)

print(f"📊 EAD Modeling Dataset Initialized. Default Observations: {n_samples}")
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

# =====================================================================
# 3. MODEL TRAINING: CCF ESTIMATION VIA REGULARIZED RIDGE REGRESSION
# =====================================================================
# Ridge regression helps to handle multicollinearity among delinquency features
model_ead = Ridge(alpha=1.0)
model_ead.fit(X_train, y_train)

# Predict CCF and enforce regulatory boundaries [0.0, 1.0]
pred_ccf_train = np.clip(model_ead.predict(X_train), 0.0, 1.0)
pred_ccf_test = np.clip(model_ead.predict(X_test), 0.0, 1.0)

print("--- CCF MODEL VALIDATION METRICS ---")
print(f"Train CCF RMSE: {np.sqrt(mean_squared_error(y_train, pred_ccf_train)):.4f}")
print(f"Test CCF RMSE : {np.sqrt(mean_squared_error(y_test, pred_ccf_test)):.4f}")
print(f"Test CCF R2   : {r2_score(y_test, pred_ccf_test):.4f}\n")

# =====================================================================
# 4. FINAL EAD RECONSTRUCTION & TESTING
# =====================================================================
# Isolate the specific test segments from the lookup dataframe
test_df = df_lkp.loc[idx_test].copy()
test_df['Predicted_CCF'] = pred_ccf_test

# Reconstruct EAD using the regulatory formula
test_df['Predicted_EAD'] = test_df['Current_Drawn'] + (test_df['Predicted_CCF'] * test_df['Unutilized_Limit'])

# Evaluate final EAD dollar/monetary error metric
true_ead_test = test_df['True_EAD']
pred_ead_test = test_df['Predicted_EAD']

print("--- FINAL INTEGRATED EAD PERFORMANCE ---")
print(f"EAD Prediction RMSE ($/Currency): {np.sqrt(mean_squared_error(true_ead_test, pred_ead_test)):.2f}")
print(f"Final EAD R2 Score              : {r2_score(true_ead_test, pred_ead_test):.4f}")

# Display operational profile of the EAD results
print("\n--- PREDICTED EAD DISTRIBUTION PROFILE (TEST) ---")
print(pred_ead_test.describe())
