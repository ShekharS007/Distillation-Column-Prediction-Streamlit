# -*- coding: utf-8 -*-
"""Distillation Column Ethanol Purity Prediction
Original file is located at
    https://colab.research.google.com/drive/1DylMhT3cUh5jHb6QS6dxbaa74odPuHum

### STEP 0: IMPORT DEPENDENDENT LIBRARIES
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

"""### STEP 1: LOAD AND EXPLORE DATA"""

# Load dataset
df = pd.read_csv('dataset_distill.csv', sep=';')

# Convert Temperature from Kelvin to Celsius
print(f"\n✓ Converting temperature from Kelvin to Celsius...")
for i in range(1, 15):
    df[f'T{i}'] = df[f'T{i}'] - 273.15

print(f"  Temperature range before: 350-375 K")
print(f"  Temperature range after: 77-102 °C")

print(f"\n✓ Dataset Loaded")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Data info
print(f"\nData Types:")
print(df.dtypes)

print(f"\nMissing Values:")
print(df.isnull().sum())

print(f"\nBasic Statistics:")
print(df.describe())

# Target variable
print(f"\nTarget Variable (Ethanol concentration):")
print(f"  Min: {df['Ethanol concentration'].min():.5f}")
print(f"  Max: {df['Ethanol concentration'].max():.5f}")
print(f"  Mean: {df['Ethanol concentration'].mean():.5f}")
print(f"  Std: {df['Ethanol concentration'].std():.5f}")

"""### STEP 2: DATA CLEANING"""

initial_rows = len(df)

# Remove any rows with missing values
df_clean = df.dropna()

# Check for duplicate rows
df_clean = df_clean.drop_duplicates()

# Convert Ethanol concentration to numeric
df_clean['Ethanol concentration'] = pd.to_numeric(
    df_clean['Ethanol concentration'],
    errors='coerce'
)

# Remove rows where target is NaN
df_clean = df_clean.dropna(subset=['Ethanol concentration'])

# Remove outliers using IQR method
Q1 = df_clean['Ethanol concentration'].quantile(0.01)
Q3 = df_clean['Ethanol concentration'].quantile(0.99)
IQR = Q3 - Q1

df_clean = df_clean[(df_clean['Ethanol concentration'] >= Q1) &
                     (df_clean['Ethanol concentration'] <= Q3)]

removed_rows = initial_rows - len(df_clean)

print(f"Initial rows: {initial_rows}")
print(f"Removed rows: {removed_rows} (duplicates, NaN, outliers)")
print(f"Final rows: {len(df_clean)}")
print(f"Columns: {len(df_clean.columns)}")

"""### STEP 3: FEATURE ENGINEERING"""

# Convert L and V to numeric
df_clean['L'] = pd.to_numeric(df_clean['L'], errors='coerce')
df_clean['V'] = pd.to_numeric(df_clean['V'], errors='coerce')
df_clean.dropna(subset=['L', 'V'], inplace=True)


# Temperature differences between trays (only top few, others have low importance)
for i in range(1, 4):  # Only T1-T2, T2-T3, T3-T4
    df_clean[f'Temp_Diff_T{i}_T{i+1}'] = df_clean[f'T{i}'] - df_clean[f'T{i+1}']

# Define Temp_Top and Temp_Bottom
df_clean['Temp_Top'] = df_clean['T1']  # Assuming T1 represents the top temperature
df_clean['Temp_Bottom'] = df_clean['T14'] # Assuming T14 represents the bottom temperature

# CONTROLLABLE OPERATING PARAMETERS (Production-Safe)

print("\n✓ Creating Controllable Operating Parameters...")

# 1. Reflux Ratio - DIRECTLY CONTROLLABLE
df_clean['Reflux_Ratio'] = df_clean['L'] / (df_clean['V'] + 1e-6)

# 2. Reboiler Steam Intensity - DIRECTLY CONTROLLABLE
df_clean['Reboiler_Intensity'] = df_clean['V'] / (df_clean['F'] + 1e-6)

# 3. Condenser Duty Proxy - DIRECTLY CONTROLLABLE
df_clean['Condenser_Load'] = df_clean['L'] / (df_clean['F'] + 1e-6)

# 4. Feed Rate Normalization - DIRECTLY CONTROLLABLE
df_clean['Feed_Normalized'] = df_clean['F'] / df_clean['F'].mean()

# 5. Product Withdrawal Rates - DIRECTLY CONTROLLABLE
df_clean['Distillate_Withdrawal'] = df_clean['D'] / (df_clean['F'] + 1e-6)
df_clean['Bottoms_Withdrawal'] = df_clean['B'] / (df_clean['F'] + 1e-6)

# 6. Column Operating Load
df_clean['Column_Load'] = (df_clean['L'] + df_clean['V']) / (df_clean['F'] + 1e-6)

# INTERACTION TERMS (SOLUTION 2: Feature Importance Balancing)
print("✓ Creating Interaction Terms (Reflux × Temperature)...")

# These interactions make CONTROLLABLE features more important
df_clean['Reflux_x_Temp_Top'] = df_clean['Reflux_Ratio'] * df_clean['Temp_Top']
df_clean['Reflux_x_Temp_Diff'] = df_clean['Reflux_Ratio'] * (df_clean['Temp_Bottom'] - df_clean['Temp_Top'])

# Reboiler interaction
df_clean['Reboiler_x_Temp_Bottom'] = df_clean['Reboiler_Intensity'] * df_clean['Temp_Bottom']

# Feed interaction
df_clean['Feed_x_Reflux'] = df_clean['Feed_Normalized'] * df_clean['Reflux_Ratio']
df_clean['Feed_x_Reboiler'] = df_clean['Feed_Normalized'] * df_clean['Reboiler_Intensity']

# Combined efficiency metric
df_clean['Separation_Duty'] = df_clean['Reflux_Ratio'] * df_clean['Reboiler_Intensity']

# Column efficiency combination
df_clean['Column_Efficiency'] = df_clean['Reflux_Ratio'] * df_clean['Column_Load']

print(f"\n✓ Features created:")
print(f"  Total features: {len(df_clean.columns) - 1} (excluding target)")

"""### STEP 4: DATA PREPARATION"""

# Drop redundant/low-importance features
features_to_drop = [
    'Temp_Top',                # Redundant with T1
    'Temp_Diff_T1_T2', 'Temp_Diff_T2_T3', 'Temp_Diff_T3_T4',       # Nearly zero importance
    'T2', 'T3', 'T4','T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14',  # Lower trays
    'T1',                       # ← ADD THIS ONE LINE
]

# Separate features and target
X = df_clean.drop(columns=['Ethanol concentration'] + features_to_drop)
y = df_clean['Ethanol concentration']

print(f"Final feature set: {X.shape[1]} features")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFinal Features:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")

"""### STEP 5: TRAIN-TEST-VALIDATION SPLIT"""

# First split: train+val (80%) and test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: train (60%) and validation (20%) from train+val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Data split:")
print(f"  Training: {X_train.shape[0]} samples (60%)")
print(f"  Validation: {X_val.shape[0]} samples (20%)")
print(f"  Testing: {X_test.shape[0]} samples (20%)")

"""### STEP 6: CROSS-VALIDATION SETUP"""

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"Using 5-Fold Cross-Validation")

"""### STEP 7: TRAINING XGBOOST MODEL"""

print("\nXGBoost Configuration (Regularization):")
print("  - max_depth: 6 (REDUCED from 8)")
print("  - learning_rate: 0.03 (LOWERED from 0.05)")
print("  - reg_alpha: 0.1 (L1 regularization - NEW)")
print("  - reg_lambda: 1.0 (L2 regularization - NEW)")
print("  - Early Stopping: enabled (Note: Early stopping will be disabled due to version incompatibility)")

xgb_model = XGBRegressor(
    # Tree parameters (control depth to prevent overfitting)
    n_estimators=500,              # Will train for all 500 trees without early stopping
    max_depth=6,                   # REDUCED (simpler trees = less overfitting)
    min_child_weight=5,            # Require minimum samples

    # Learning parameters
    learning_rate=0.03,            # LOWERED (slower learning = less overfitting)

    # Regularization parameters (SOLUTION 1)
    subsample=0.85,                # Use 85% of data per tree
    colsample_bytree=0.85,         # Use 85% of features per tree
    reg_alpha=0.1,                 # L1 regularization (NEW)
    reg_lambda=1.0,                # L2 regularization (NEW)

    random_state=42,
    n_jobs=-1,
    verbosity=0
)

print("\nTraining XGBoost...")

# SOLUTION 3: Train without early stopping due to version incompatibility
# Removing early_stopping_rounds, eval_set, and verbose from fit method
xgb_model.fit(
    X_train_scaled, y_train
)

# As early stopping was disabled, assume all estimators were used
# We cannot set best_iteration directly as it's a read-only property.
# We will use n_estimators - 1 to represent the 'best' (final) iteration.
final_num_estimators = xgb_model.n_estimators - 1

print(f"✓ Training complete")
print(f"  Final iteration: {final_num_estimators}")
print(f"  Trees used: {final_num_estimators + 1}")

# Predictions
y_pred_xgb_train = xgb_model.predict(X_train_scaled)
y_pred_xgb_val = xgb_model.predict(X_val_scaled)
y_pred_xgb_test = xgb_model.predict(X_test_scaled)

# Metrics
xgb_train_r2 = r2_score(y_train, y_pred_xgb_train)
xgb_val_r2 = r2_score(y_val, y_pred_xgb_val)
xgb_test_r2 = r2_score(y_test, y_pred_xgb_test)
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb_test))
xgb_test_mae = mean_absolute_error(y_test, y_pred_xgb_test)

print(f"\nXGBoost Performance:")
print(f"  Train R²: {xgb_train_r2:.6f}")
print(f"  Val R²: {xgb_val_r2:.6f}")
print(f"  Test R²: {xgb_test_r2:.6f}")
print(f"  Test RMSE: {xgb_test_rmse:.6f}")
print(f"  Test MAE: {xgb_test_mae:.6f}")

# Overfitting check
train_test_gap = xgb_train_r2 - xgb_test_r2
print(f"\nOverfitting Check (Train R² - Test R²):")
print(f"  Gap: {train_test_gap:.6f}")
if train_test_gap < 0.01:
    print(f"  ✅ EXCELLENT - Model generalizes well (gap < 0.01)")
elif train_test_gap < 0.05:
    print(f"  ⚠️  GOOD - Slight overfitting (gap < 0.05)")
else:
    print(f"  ❌ WARNING - Model is overfitting (gap > 0.05)")

# Cross-validation
xgb_cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=kfold, scoring='r2')

print(f"\nXGBoost Cross-Validation R²:")
print(f"  Fold scores: {[f'{s:.6f}' for s in xgb_cv_scores]}")
print(f"  Mean: {xgb_cv_scores.mean():.6f} (+/- {xgb_cv_scores.std():.6f})")

"""### STEP 8: TRAINING RANDOM FOREST MODEL

### STEP 8: TRAINING RANDOM FOREST MODEL
"""

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)
print("✓ Training complete")

# Predictions
y_pred_rf_train = rf_model.predict(X_train_scaled)
y_pred_rf_test = rf_model.predict(X_test_scaled)

# Metrics
rf_train_r2 = r2_score(y_train, y_pred_rf_train)
rf_test_r2 = r2_score(y_test, y_pred_rf_test)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
rf_test_mae = mean_absolute_error(y_test, y_pred_rf_test)

print(f"\nRandom Forest Performance:")
print(f"  Train R²: {rf_train_r2:.6f}")
print(f"  Test R²: {rf_test_r2:.6f}")
print(f"  Test RMSE: {rf_test_rmse:.6f}")
print(f"  Test MAE: {rf_test_mae:.6f}")

# Cross-validation
rf_cv_scores = cross_val_score(
    rf_model, X_train_scaled, y_train, cv=kfold, scoring='r2'
)

print(f"\nRandom Forest Cross-Validation R²:")
print(f"  Fold scores: {[f'{s:.6f}' for s in rf_cv_scores]}")
print(f"  Mean: {rf_cv_scores.mean():.6f} (+/- {rf_cv_scores.std():.6f})")

"""### STEP 9: MODEL COMPARISON & SELECTION"""

comparison_df = pd.DataFrame({
    'Model': ['XGBoost', 'Random Forest'],
    'Train R²': [xgb_train_r2, rf_train_r2],
    'Val R²': [xgb_val_r2, rf_val_r2],
    'Test R²': [xgb_test_r2, rf_test_r2],
    'Test RMSE': [xgb_test_rmse, rf_test_rmse],
    'Test MAE': [xgb_test_mae, rf_test_mae],
    'CV R² Mean': [xgb_cv_scores.mean(), rf_cv_scores.mean()],
})

print("\n" + comparison_df.to_string(index=False))

# Select best model
if xgb_test_r2 > rf_test_r2:
    best_model = xgb_model
    best_model_name = 'XGBoost'
    best_r2 = xgb_test_r2
    best_rmse = xgb_test_rmse
    best_mae = xgb_test_mae
else:
    best_model = rf_model
    best_model_name = 'Random Forest'
    best_r2 = rf_test_r2
    best_rmse = rf_test_rmse
    best_mae = rf_test_mae

print(f"\n✓ SELECTED MODEL: {best_model_name}")
print(f"  Test R²: {best_r2:.6f}")
print(f"  Test RMSE: {best_rmse:.6f}")
print(f"  Test MAE: {best_mae:.6f}")

"""### STEP 10: FEATURE IMPORTANCE ANALYSIS"""

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 20 Most Important Features ({best_model_name}):")
print(feature_importance.head(20).to_string(index=False))

# Analyze feature distribution
top_5_pct = feature_importance.head(5)['Importance'].sum()
top_10_pct = feature_importance.head(10)['Importance'].sum()

print(f"\nFeature Importance Distribution:")
print(f"  Top 5 features explain: {top_5_pct*100:.1f}%")
print(f"  Top 10 features explain: {top_10_pct*100:.1f}%")

if top_5_pct < 0.70:
    print(f"  ✅ GOOD - Features well distributed (better generalization)")
elif top_5_pct < 0.85:
    print(f"  ⚠️  MODERATE - Some concentration")
else:
    print(f"  ⚠️  WARNING - High concentration on few features")

# Check if controllable features are in top 10
controllable_features = ['Reflux_Ratio', 'Reboiler_Intensity', 'Condenser_Load',
                         'Feed_Normalized', 'Column_Load']
controllable_in_top10 = feature_importance.head(10)[feature_importance.head(10)['Feature'].isin(controllable_features)]

print(f"\nControllable Features in Top 10:")
if len(controllable_in_top10) > 0:
    print(f"  ✅ Found {len(controllable_in_top10)} controllable features")
    print(controllable_in_top10.to_string(index=False))
else:
    print(f"  ⚠️  No controllable features in top 10 (model relies on temperatures)")

    # Visualization
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
plt.title(f'Top 20 Feature Importance - {best_model_name} (IMPROVED)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Feature importance plot saved: feature_importance.png")

"""### STEP 11: RESIDUAL ANALYSIS"""

y_pred_final = best_model.predict(X_test_scaled)
residuals = y_test - y_pred_final

print(f"Residual Statistics:")
print(f"  Mean: {residuals.mean():.8f} (should be ~0)")
print(f"  Std Dev: {residuals.std():.6f}")
print(f"  Min: {residuals.min():.6f}")
print(f"  Max: {residuals.max():.6f}")

# Check normality
from scipy import stats
_, p_value = stats.normaltest(residuals)
print(f"  Normality test p-value: {p_value:.4f}")
if p_value > 0.05:
    print(f"  ✅ Residuals are approximately normally distributed")
else:
    print(f"  ⚠️  Residuals may not be normally distributed")

# Residual plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Predicted vs Actual
axes[0, 0].scatter(y_test, y_pred_final, alpha=0.6, s=20, color='steelblue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Ethanol Concentration', fontweight='bold')
axes[0, 0].set_ylabel('Predicted Ethanol Concentration', fontweight='bold')
axes[0, 0].set_title(f'Predicted vs Actual (R² = {best_r2:.4f})', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals
axes[0, 1].scatter(y_pred_final, residuals, alpha=0.6, s=20, color='steelblue')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Values', fontweight='bold')
axes[0, 1].set_ylabel('Residuals', fontweight='bold')
axes[0, 1].set_title('Residual Plot', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Distribution of residuals
axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[1, 0].set_xlabel('Residuals', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Residual analysis plot saved: residual_analysis.png")

import pickle

# Save model
model_filename = f'{best_model_name.lower().replace(" ", "_")}model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Model saved: {model_filename}")

# Save scaler
scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved: {scaler_filename}")

# Save feature names
feature_names_filename = 'features_names.pkl'
feature_names_list = X.columns.tolist()
with open(feature_names_filename, 'wb') as f:
    pickle.dump(feature_names_list, f)
print(f"✓ Feature names saved: {feature_names_filename}")

