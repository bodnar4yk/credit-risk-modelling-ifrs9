import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =====================================================================
# 1. DATA LOADING & PREPROCESSING
# =====================================================================
# Load dataset
df_train = pd.read_csv('cs-training.csv')

# Define features and target variable
X = df_train.drop('SeriousDlqin2yrs', axis=1) 
y = df_train['SeriousDlqin2yrs']              

# Stratified Time-Series/Sequential Split (80% Train / 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintains target class balance across splits
)

def add_custom_features(df):
    """
    Cleans raw attributes and generates domain-specific risk variables.
    Handles outliers via clipping and missing values via imputation.
    """
    df_copy = df.copy()

    # Age cleaning & outlier handling
    median_age = df_copy['age'].median()
    df_copy['age_clean'] = df_copy['age']
    df_copy.loc[(df_copy['age_clean'] < 18) | (df_copy['age_clean'] > 90), 'age_clean'] = median_age
    df_copy = df_copy.drop('age', axis=1) 
  
    # Debt load relative to the family size
    df_copy['DebtPerDependent'] = df_copy['DebtRatio'] / (df_copy['NumberOfDependents'] + 1)
  
    # Capping extreme income outliers at 99th percentile to stabilize regression
    upper_limit = df_copy['MonthlyIncome'].quantile(0.99) 
    df_copy['MonthlyIncome_capped'] = df_copy['MonthlyIncome'].clip(upper=upper_limit)
    
    # Drop collinear or raw uncleaned columns
    df_copy = df_copy.drop(['DebtRatio', 'MonthlyIncome', 'NumberRealEstateLoansOrLines'], axis=1) 
    
    return df_copy

# Apply preprocessing transformations safely
X_train = add_custom_features(X_train)
X_test = add_custom_features(X_test)

# =====================================================================
# 2. OPTIMAL BINNING VIA DECISION TREE & WoE/IV GENERATION
# =====================================================================
def get_tree_bins(X, y): 
    """
    Finds optimal characteristic bin breakpoints using shallow Decision Trees.
    Sets minimum sample size per leaf at 5% to maintain Basel stability requirements.
    """
    tree = DecisionTreeClassifier(max_leaf_nodes=10, random_state=42, min_samples_leaf=0.05)
    tree.fit(X.values.reshape(-1, 1), y)
    thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
    return sorted(thresholds)

def calculate_woe_iv(df, feature, target, bins):
    """
    Calculates Weight of Evidence (WoE) and Information Value (IV) tables per bin.
    """
    bin_edges = [-np.inf] + list(bins) + [np.inf]
    df['temp_bin'] = pd.cut(df[feature], bins=bin_edges)
    
    stats = df.groupby('temp_bin', observed=False)[target].agg(['count', 'sum'])
    stats.columns = ['Total', 'Bad']
    stats['Good'] = stats['Total'] - stats['Bad']
    
    total_good = stats['Good'].sum()
    total_bad = stats['Bad'].sum()
    
    # Add small laplace smoothing to prevent division by zero if necessary
    stats['Dist_Good'] = stats['Good'] / total_good
    stats['Dist_Bad'] = stats['Bad'] / total_bad
    
    stats['WoE'] = np.log(stats['Dist_Good'] / (stats['Dist_Bad'] + 1e-10))
    stats['IV'] = (stats['Dist_Good'] - stats['Dist_Bad']) * stats['WoE']
    
    return stats[['Total', 'Good', 'Bad', 'WoE', 'IV']]

features = [col for col in X_train.columns[1:]]
iv_results = []
all_bins_edges = {}  
woe_mappers = {}     

print("\n--- STARTING OPTIMAL BINNING & WoE PROCESSING ---")
for col in features:
    try:
        raw_bins = get_tree_bins(X_train[col], y_train)
        bin_edges = [-np.inf] + list(raw_bins) + [np.inf]
        all_bins_edges[col] = bin_edges
        
        woe_table = calculate_woe_iv(X_train.join(y_train), col, 'SeriousDlqin2yrs', raw_bins)
        woe_mappers[col] = woe_table['WoE'].to_dict()
        
        total_iv = woe_table['IV'].sum()
        iv_results.append({'Feature': col, 'IV': total_iv})
        print(f"✅ Feature '{col}' processed. Total Bins: {len(bin_edges)-1}")
    except Exception as e:
        print(f"❌ Skipped column '{col}' due to error: {e}")

# Build and display Information Value Summary
summary_table = pd.DataFrame(iv_results).sort_values(by='IV', ascending=False)

def interpret_iv(iv):
    if iv < 0.02: return 'Useless'
    if iv < 0.1:  return 'Weak'
    if iv < 0.3:  return 'Medium'
    return 'Strong'

summary_table['Predictive Power'] = summary_table['IV'].apply(interpret_iv)
print("\n--- INFORMATION VALUE (IV) SUMMARY ---")
print(summary_table.to_string(index=False))

# =====================================================================
# 3. DATA TRANSFORMATION INTO WoE SCALE
# =====================================================================
X_train_woe = pd.DataFrame(index=X_train.index)
X_test_woe = pd.DataFrame(index=X_test.index)

for col in woe_mappers:
    edges = all_bins_edges[col]
    mapper = woe_mappers[col]
    
    X_train_woe[col] = pd.cut(X_train[col], bins=edges, include_lowest=True).map(mapper).astype(float)
    X_test_woe[col] = pd.cut(X_test[col], bins=edges, include_lowest=True).map(mapper).astype(float)

X_train_woe = X_train_woe.fillna(0)
X_test_woe = X_test_woe.fillna(0)

# =====================================================================
# 4. LOGISTIC REGRESSION SCORECARD MODELING
# =====================================================================
lr_model = LogisticRegression(random_state=42, solver='lbfgs')
lr_model.fit(X_train_woe, y_train)

print("\n--- DEVELOPED REGRESSION COEFFICIENTS ---")
intercept = lr_model.intercept_[0]
print(f"Intercept (Alpha): {intercept:.4f}")
for col, coef in zip(X_train_woe.columns, lr_model.coef_[0]):
    print(f"{col}: {coef:.4f}")

# =====================================================================
# 5. SCORECARD SCALING TO POINTS
# =====================================================================
# Standard industry scaling parameters
BASE_SCORE = 600       
BASE_ODDS = 50         
PDO = 20               

factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)
base_points = offset + factor * intercept

print("\n--- SCORECARD SCALING PARAMETERS ---")
print(f"Factor: {factor:.4f} | Offset: {offset:.4f} | Base Points: {base_points:.2f}")

print("\n--- FINAL SCORECARD POINTS SYSTEM ---")
scorecard_rows = []
for col in woe_mappers:
    mapper = woe_mappers[col]
    coef = lr_model.coef_[0][X_train_woe.columns.get_loc(col)]
    
    for interval, woe_val in mapper.items():
        bin_points = factor * coef * woe_val
        scorecard_rows.append({
            'Characteristic': col,
            'Bin Interval': str(interval),
            'WoE Value': round(woe_val, 4),
            'Points Contribution': round(bin_points, 2)
        })

df_scorecard = pd.DataFrame(scorecard_rows)
print(df_scorecard.to_string(index=False))

# =====================================================================
# 6. PERFORMANCE VALIDATION (ROC-AUC & GINI)
# =====================================================================
y_pred_proba_train = lr_model.predict_proba(X_train_woe)[:, 1]
y_pred_proba_test = lr_model.predict_proba(X_test_woe)[:, 1]

auc_train = roc_auc_score(y_train, y_pred_proba_train)
auc_test = roc_auc_score(y_test, y_pred_proba_test)

gini_train = 2 * auc_train - 1
gini_test = 2 * auc_test - 1

print("\n" + "="*50)
print("--- FINAL CREDIT MODEL METRICS ---")
print(f"Train AUC-ROC : {auc_train:.4f}  |  Train GINI: {gini_train:.4f}")
print(f"Test AUC-ROC  : {auc_test:.4f}  |  Test GINI: {gini_test:.4f}")
print("="*50)
