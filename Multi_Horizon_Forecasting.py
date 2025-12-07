"""
Multi-Horizon Forecasting Analysis
Evaluating 1, 3, 6, and 12-month ahead forecasts for all models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MULTI-HORIZON FORECASTING ANALYSIS")
print("Evaluating 1, 3, 6, and 12-month ahead forecasts")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'VehicleData-1.csv')
df = pd.read_csv(data_path)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')
df.columns = ['Total_Sales', 'New_Orders']

# Split data
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

print(f"\nDataset: {len(df)} observations")
print(f"Training: {len(train)} observations")
print(f"Test: {len(test)} observations")

# ============================================================================
# 2. MULTI-HORIZON FORECAST FUNCTION
# ============================================================================

def evaluate_multi_horizon(model_name, predictions_dict, actual, horizons=[1, 3, 6, 12]):
    """
    Evaluate forecasts at multiple horizons
    predictions_dict: {horizon: predictions_array}
    """
    results = []
    
    for h in horizons:
        if h > len(actual):
            continue
            
        pred = predictions_dict[h]
        act = actual[:len(pred)]
        
        rmse = np.sqrt(mean_squared_error(act, pred))
        mae = mean_absolute_error(act, pred)
        mape = mean_absolute_percentage_error(act, pred) * 100
        
        results.append({
            'Model': model_name,
            'Horizon': h,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        })
    
    return results

# ============================================================================
# 3. ARIMA MULTI-HORIZON FORECASTS
# ============================================================================
print("\n" + "="*80)
print("ARIMA(3,1,1) MULTI-HORIZON FORECASTS")
print("="*80)

arima_model = ARIMA(train['Total_Sales'], order=(3,1,1))
arima_fitted = arima_model.fit()

arima_predictions = {}
for h in [1, 3, 6, 12]:
    if h <= len(test):
        arima_predictions[h] = arima_fitted.forecast(steps=h)

arima_results = evaluate_multi_horizon('ARIMA(3,1,1)', arima_predictions, test['Total_Sales'])

for r in arima_results:
    print(f"Horizon {r['Horizon']}-month: RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}, MAPE={r['MAPE']:.2f}%")

# ============================================================================
# 4. SARIMA MULTI-HORIZON FORECASTS
# ============================================================================
print("\n" + "="*80)
print("SARIMA(1,1,1)(2,1,1,12) MULTI-HORIZON FORECASTS")
print("="*80)

sarima_model = SARIMAX(train['Total_Sales'], order=(1,1,1), seasonal_order=(2,1,1,12))
sarima_fitted = sarima_model.fit(disp=False)

sarima_predictions = {}
for h in [1, 3, 6, 12]:
    if h <= len(test):
        sarima_predictions[h] = sarima_fitted.forecast(steps=h)

sarima_results = evaluate_multi_horizon('SARIMA(1,1,1)(2,1,1,12)', sarima_predictions, test['Total_Sales'])

for r in sarima_results:
    print(f"Horizon {r['Horizon']}-month: RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}, MAPE={r['MAPE']:.2f}%")

# ============================================================================
# 5. VAR MULTI-HORIZON FORECASTS
# ============================================================================
print("\n" + "="*80)
print("VAR(8) MULTI-HORIZON FORECASTS")
print("="*80)

# Difference the data
df_diff = df.diff().dropna()
train_diff = df_diff[:train_size-1]
test_diff = df_diff[train_size-1:]

var_model = VAR(train_diff)
var_fitted = var_model.fit(8)

var_predictions = {}
for h in [1, 3, 6, 12]:
    if h <= len(test_diff):
        forecast_diff = var_fitted.forecast(train_diff.values[-8:], steps=h)
        
        # Convert back to levels
        forecast_levels = []
        last_level = df.iloc[train_size-1]['Total_Sales']
        
        for i in range(h):
            if i == 0:
                forecast_levels.append(last_level + forecast_diff[i, 0])
            else:
                forecast_levels.append(forecast_levels[i-1] + forecast_diff[i, 0])
        
        var_predictions[h] = np.array(forecast_levels)

var_results = evaluate_multi_horizon('VAR(8)', var_predictions, test['Total_Sales'])

for r in var_results:
    print(f"Horizon {r['Horizon']}-month: RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}, MAPE={r['MAPE']:.2f}%")

# ============================================================================
# 6. MACHINE LEARNING MULTI-HORIZON FORECASTS
# ============================================================================
print("\n" + "="*80)
print("MACHINE LEARNING MULTI-HORIZON FORECASTS")
print("="*80)

# Feature engineering function
def create_features(data, target_col='Total_Sales', lags=[1, 2, 3, 6, 12], 
                    rolling_windows=[3, 6, 12]):
    df_features = data.copy()
    
    for lag in lags:
        df_features[f'{target_col}_lag_{lag}'] = df_features[target_col].shift(lag)
    
    for window in rolling_windows:
        df_features[f'{target_col}_rolling_mean_{window}'] = df_features[target_col].rolling(window=window).mean()
        df_features[f'{target_col}_rolling_std_{window}'] = df_features[target_col].rolling(window=window).std()
        df_features[f'{target_col}_rolling_min_{window}'] = df_features[target_col].rolling(window=window).min()
        df_features[f'{target_col}_rolling_max_{window}'] = df_features[target_col].rolling(window=window).max()
    
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['year'] = df_features.index.year
    df_features['day_of_year'] = df_features.index.dayofyear
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features[f'{target_col}_diff_1'] = df_features[target_col].diff(1)
    df_features[f'{target_col}_diff_12'] = df_features[target_col].diff(12)
    
    if 'New_Orders' in df_features.columns:
        for lag in [1, 2, 3]:
            df_features[f'New_Orders_lag_{lag}'] = df_features['New_Orders'].shift(lag)
        df_features['New_Orders_rolling_mean_3'] = df_features['New_Orders'].rolling(window=3).mean()
    
    return df_features

# Prepare data
df_ml = create_features(df, target_col='Total_Sales')
df_ml_clean = df_ml.dropna()

target = 'Total_Sales'
feature_cols = [col for col in df_ml_clean.columns if col not in [target, 'New_Orders']]

X = df_ml_clean[feature_cols]
y = df_ml_clean[target]

train_size_ml = int(len(X) * 0.8)
X_train, X_test = X[:train_size_ml], X[train_size_ml:]
y_train, y_test = y[:train_size_ml], y[train_size_ml:]

# Train models
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

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

# Multi-horizon predictions for ML
xgb_predictions = {}
lgb_predictions = {}

for h in [1, 3, 6, 12]:
    if h <= len(X_test):
        xgb_predictions[h] = xgb_model.predict(X_test[:h])
        lgb_predictions[h] = lgb_model.predict(X_test[:h])

xgb_results = evaluate_multi_horizon('XGBoost', xgb_predictions, y_test)
lgb_results = evaluate_multi_horizon('LightGBM', lgb_predictions, y_test)

print("\nXGBoost:")
for r in xgb_results:
    print(f"Horizon {r['Horizon']}-month: RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}, MAPE={r['MAPE']:.2f}%")

print("\nLightGBM:")
for r in lgb_results:
    print(f"Horizon {r['Horizon']}-month: RMSE={r['RMSE']:.4f}, MAE={r['MAE']:.4f}, MAPE={r['MAPE']:.2f}%")

# ============================================================================
# 7. COMPILE ALL RESULTS
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE MULTI-HORIZON RESULTS")
print("="*80)

all_results = arima_results + sarima_results + var_results + xgb_results + lgb_results
results_df = pd.DataFrame(all_results)

print("\n" + results_df.to_string(index=False))

# ============================================================================
# 8. VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Multi-Horizon Forecast Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['RMSE', 'MAE', 'MAPE']
horizons = [1, 3, 6, 12]

# Plot 1: RMSE by Horizon
ax1 = axes[0, 0]
for model in results_df['Model'].unique():
    model_data = results_df[results_df['Model'] == model]
    ax1.plot(model_data['Horizon'], model_data['RMSE'], marker='o', linewidth=2, label=model)
ax1.set_xlabel('Forecast Horizon (months)', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_title('RMSE by Forecast Horizon', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(horizons)

# Plot 2: MAE by Horizon
ax2 = axes[0, 1]
for model in results_df['Model'].unique():
    model_data = results_df[results_df['Model'] == model]
    ax2.plot(model_data['Horizon'], model_data['MAE'], marker='o', linewidth=2, label=model)
ax2.set_xlabel('Forecast Horizon (months)', fontsize=12)
ax2.set_ylabel('MAE', fontsize=12)
ax2.set_title('MAE by Forecast Horizon', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(horizons)

# Plot 3: MAPE by Horizon
ax3 = axes[1, 0]
for model in results_df['Model'].unique():
    model_data = results_df[results_df['Model'] == model]
    ax3.plot(model_data['Horizon'], model_data['MAPE'], marker='o', linewidth=2, label=model)
ax3.set_xlabel('Forecast Horizon (months)', fontsize=12)
ax3.set_ylabel('MAPE (%)', fontsize=12)
ax3.set_title('MAPE by Forecast Horizon', fontsize=14, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(horizons)

# Plot 4: Heatmap of RMSE
ax4 = axes[1, 1]
pivot_rmse = results_df.pivot(index='Model', columns='Horizon', values='RMSE')
sns.heatmap(pivot_rmse, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax4, cbar_kws={'label': 'RMSE'})
ax4.set_title('RMSE Heatmap: Model vs Horizon', fontsize=14, fontweight='bold')
ax4.set_xlabel('Forecast Horizon (months)', fontsize=12)
ax4.set_ylabel('Model', fontsize=12)

plt.tight_layout()
output_path = os.path.join(script_dir, 'outputs', 'multi_horizon_forecasts.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nMulti-horizon forecast plot saved as 'outputs/multi_horizon_forecasts.png'")

# ============================================================================
# 9. SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: BEST MODEL BY HORIZON")
print("="*80)

for h in horizons:
    horizon_data = results_df[results_df['Horizon'] == h]
    best_rmse = horizon_data.loc[horizon_data['RMSE'].idxmin()]
    print(f"\n{h}-Month Horizon:")
    print(f"  Best Model: {best_rmse['Model']}")
    print(f"  RMSE: {best_rmse['RMSE']:.4f}")
    print(f"  MAE: {best_rmse['MAE']:.4f}")
    print(f"  MAPE: {best_rmse['MAPE']:.2f}%")

print("\n" + "="*80)
print("MULTI-HORIZON ANALYSIS COMPLETE!")
print("="*80)

# Save results to CSV
output_csv = os.path.join(script_dir, 'outputs', 'multi_horizon_results.csv')
results_df.to_csv(output_csv, index=False)
print(f"\nResults saved to 'outputs/multi_horizon_results.csv'")
