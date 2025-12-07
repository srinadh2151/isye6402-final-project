"""
Machine Learning Analysis: Gradient Boosting Methods
XGBoost and LightGBM for Vehicle Sales Forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MACHINE LEARNING ANALYSIS: GRADIENT BOOSTING METHODS")
print("XGBoost and LightGBM for Vehicle Sales Forecasting")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("1. DATA LOADING AND FEATURE ENGINEERING")
print("="*80)

# Load data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'VehicleData-1.csv')
df = pd.read_csv(data_path)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')
df.columns = ['Total_Sales', 'New_Orders']

print(f"\nDataset Shape: {df.shape}")
print(f"Date Range: {df.index.min()} to {df.index.max()}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("2. FEATURE ENGINEERING")
print("="*80)

def create_features(data, target_col='Total_Sales', lags=[1, 2, 3, 6, 12], 
                    rolling_windows=[3, 6, 12]):
    """
    Create time series features for machine learning
    """
    df_features = data.copy()
    
    # Lag features
    for lag in lags:
        df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Rolling statistics
    for window in rolling_windows:
        df_features[f'{target_col}_rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
        df_features[f'{target_col}_rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
        df_features[f'{target_col}_rolling_min_{window}'] = df_features[target_col].rolling(window=window).min()
        df_features[f'{target_col}_rolling_max_{window}'] = df_features[target_col].rolling(window=window).max()
    
    # Time-based features
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['year'] = df_features.index.year
    df_features['day_of_year'] = df_features.index.dayofyear
    
    # Cyclical encoding for month
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Difference features
    df_features[f'{target_col}_diff_1'] = df_features[target_col].diff(1)
    df_features[f'{target_col}_diff_12'] = df_features[target_col].diff(12)
    
    # New Orders features (exogenous variable)
    if 'New_Orders' in df_features.columns:
        for lag in [1, 2, 3]:
            df_features[f'New_Orders_lag_{lag}'] = df_features['New_Orders'].shift(lag)
        df_features['New_Orders_rolling_mean_3'] = df_features['New_Orders'].rolling(window=3).mean()
    
    return df_features

# Create features
df_ml = create_features(df, target_col='Total_Sales')

# Remove rows with NaN values (due to lagging and rolling)
df_ml_clean = df_ml.dropna()

print(f"\nOriginal features: {df.shape[1]}")
print(f"Engineered features: {df_ml_clean.shape[1]}")
print(f"Samples after cleaning: {df_ml_clean.shape[0]}")

# Prepare features and target
target = 'Total_Sales'
feature_cols = [col for col in df_ml_clean.columns if col not in [target, 'New_Orders']]

X = df_ml_clean[feature_cols]
y = df_ml_clean[target]

print(f"\nFeature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("3. TRAIN-TEST SPLIT (Time Series)")
print("="*80)

# Time series split (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nTraining set: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})")
print(f"Test set: {len(X_test)} samples ({X_test.index[0]} to {X_test.index[-1]})")

# ============================================================================
# 4. XGBOOST MODEL
# ============================================================================
print("\n" + "="*80)
print("4. XGBOOST MODEL")
print("="*80)

print("\nTraining XGBoost model...")

# XGBoost parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# Train XGBoost
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)],
              verbose=False)

# Predictions
xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

# Metrics
xgb_train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
xgb_test_mae = mean_absolute_error(y_test, xgb_test_pred)
xgb_test_mape = mean_absolute_percentage_error(y_test, xgb_test_pred) * 100
xgb_test_r2 = r2_score(y_test, xgb_test_pred)

print(f"\nXGBoost Performance:")
print(f"  Training RMSE: {xgb_train_rmse:.4f}")
print(f"  Test RMSE: {xgb_test_rmse:.4f}")
print(f"  Test MAE: {xgb_test_mae:.4f}")
print(f"  Test MAPE: {xgb_test_mape:.2f}%")
print(f"  Test R²: {xgb_test_r2:.4f}")

# Feature importance
xgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features (XGBoost):")
print(xgb_importance.head(10).to_string(index=False))

# ============================================================================
# 5. LIGHTGBM MODEL
# ============================================================================
print("\n" + "="*80)
print("5. LIGHTGBM MODEL")
print("="*80)

print("\nTraining LightGBM model...")

# LightGBM parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Train LightGBM
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

# Predictions
lgb_train_pred = lgb_model.predict(X_train)
lgb_test_pred = lgb_model.predict(X_test)

# Metrics
lgb_train_rmse = np.sqrt(mean_squared_error(y_train, lgb_train_pred))
lgb_test_rmse = np.sqrt(mean_squared_error(y_test, lgb_test_pred))
lgb_test_mae = mean_absolute_error(y_test, lgb_test_pred)
lgb_test_mape = mean_absolute_percentage_error(y_test, lgb_test_pred) * 100
lgb_test_r2 = r2_score(y_test, lgb_test_pred)

print(f"\nLightGBM Performance:")
print(f"  Training RMSE: {lgb_train_rmse:.4f}")
print(f"  Test RMSE: {lgb_test_rmse:.4f}")
print(f"  Test MAE: {lgb_test_mae:.4f}")
print(f"  Test MAPE: {lgb_test_mape:.2f}%")
print(f"  Test R²: {lgb_test_r2:.4f}")

# Feature importance
lgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Most Important Features (LightGBM):")
print(lgb_importance.head(10).to_string(index=False))

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("6. MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['XGBoost', 'LightGBM'],
    'Train RMSE': [xgb_train_rmse, lgb_train_rmse],
    'Test RMSE': [xgb_test_rmse, lgb_test_rmse],
    'Test MAE': [xgb_test_mae, lgb_test_mae],
    'Test MAPE (%)': [xgb_test_mape, lgb_test_mape],
    'Test R²': [xgb_test_r2, lgb_test_r2]
})

print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# 7. VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("7. GENERATING VISUALIZATIONS")
print("="*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Plot 1: XGBoost Predictions
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(y_train.index, y_train, label='Training Data', color='blue', linewidth=2, alpha=0.7)
ax1.plot(y_test.index, y_test, label='Actual Test Data', color='green', linewidth=2)
ax1.plot(y_test.index, xgb_test_pred, label='XGBoost Predictions', 
         color='red', linewidth=2, linestyle='--')
ax1.set_title(f'XGBoost Predictions (RMSE: {xgb_test_rmse:.4f}, R²: {xgb_test_r2:.4f})', 
              fontsize=14, fontweight='bold')
ax1.set_ylabel('Total Sales (Millions)', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: LightGBM Predictions
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(y_train.index, y_train, label='Training Data', color='blue', linewidth=2, alpha=0.7)
ax2.plot(y_test.index, y_test, label='Actual Test Data', color='green', linewidth=2)
ax2.plot(y_test.index, lgb_test_pred, label='LightGBM Predictions', 
         color='orange', linewidth=2, linestyle='--')
ax2.set_title(f'LightGBM Predictions (RMSE: {lgb_test_rmse:.4f}, R²: {lgb_test_r2:.4f})', 
              fontsize=14, fontweight='bold')
ax2.set_ylabel('Total Sales (Millions)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: XGBoost Feature Importance
ax3 = fig.add_subplot(gs[2, 0])
top_features_xgb = xgb_importance.head(15)
ax3.barh(range(len(top_features_xgb)), top_features_xgb['importance'], color='steelblue')
ax3.set_yticks(range(len(top_features_xgb)))
ax3.set_yticklabels(top_features_xgb['feature'], fontsize=9)
ax3.set_xlabel('Importance', fontsize=10)
ax3.set_title('XGBoost Feature Importance (Top 15)', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: LightGBM Feature Importance
ax4 = fig.add_subplot(gs[2, 1])
top_features_lgb = lgb_importance.head(15)
ax4.barh(range(len(top_features_lgb)), top_features_lgb['importance'], color='darkorange')
ax4.set_yticks(range(len(top_features_lgb)))
ax4.set_yticklabels(top_features_lgb['feature'], fontsize=9)
ax4.set_xlabel('Importance', fontsize=10)
ax4.set_title('LightGBM Feature Importance (Top 15)', fontsize=12, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(True, alpha=0.3, axis='x')

# Plot 5: Residuals Comparison
ax5 = fig.add_subplot(gs[3, 0])
xgb_residuals = y_test - xgb_test_pred
lgb_residuals = y_test - lgb_test_pred
ax5.scatter(y_test, xgb_residuals, alpha=0.6, label='XGBoost', s=50)
ax5.scatter(y_test, lgb_residuals, alpha=0.6, label='LightGBM', s=50)
ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax5.set_xlabel('Actual Values', fontsize=10)
ax5.set_ylabel('Residuals', fontsize=10)
ax5.set_title('Residuals Plot', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Model Comparison Bar Chart
ax6 = fig.add_subplot(gs[3, 1])
metrics = ['Test RMSE', 'Test MAE', 'Test MAPE (%)']
xgb_metrics = [xgb_test_rmse, xgb_test_mae, xgb_test_mape]
lgb_metrics = [lgb_test_rmse, lgb_test_mae, lgb_test_mape]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax6.bar(x - width/2, xgb_metrics, width, label='XGBoost', color='steelblue')
bars2 = ax6.bar(x + width/2, lgb_metrics, width, label='LightGBM', color='darkorange')

ax6.set_ylabel('Value', fontsize=10)
ax6.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics, fontsize=9)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Machine Learning Analysis: Gradient Boosting Methods', 
             fontsize=16, fontweight='bold', y=0.995)

output_path = os.path.join(script_dir, 'ml_gradient_boosting.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\nML gradient boosting plots saved as 'ml_gradient_boosting.png'")

print("\n" + "="*80)
print("MACHINE LEARNING ANALYSIS COMPLETE!")
print("="*80)

print(f"\nKey Findings:")
print(f"  • XGBoost Test RMSE: {xgb_test_rmse:.4f}")
print(f"  • LightGBM Test RMSE: {lgb_test_rmse:.4f}")
print(f"  • Best Model: {'XGBoost' if xgb_test_rmse < lgb_test_rmse else 'LightGBM'}")
print(f"  • Most Important Feature (XGBoost): {xgb_importance.iloc[0]['feature']}")
print(f"  • Most Important Feature (LightGBM): {lgb_importance.iloc[0]['feature']}")
