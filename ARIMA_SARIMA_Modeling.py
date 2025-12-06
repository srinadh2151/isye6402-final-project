"""
ARIMA and SARIMA Modeling for Vehicle Sales Forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ARIMA AND SARIMA MODELING")
print("="*80)

# Load data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'VehicleData-1.csv')
df = pd.read_csv(data_path)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')
df.columns = ['Total_Sales', 'New_Orders']

# Split data: Train (80%) and Test (20%)
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

print(f"\nData Split:")
print(f"  Training set: {len(train)} observations ({train.index[0]} to {train.index[-1]})")
print(f"  Test set: {len(test)} observations ({test.index[0]} to {test.index[-1]})")

# ============================================================================
# ARIMA MODELS FOR TOTAL SALES
# ============================================================================
print("\n" + "="*80)
print("ARIMA MODELS FOR TOTAL SALES")
print("="*80)

# Test different ARIMA configurations
arima_configs = [
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
    (3, 1, 1),
    (1, 1, 3)
]

best_aic = np.inf
best_arima_order = None
arima_results = []

print("\nTesting ARIMA configurations:")
print("-" * 60)

for order in arima_configs:
    try:
        model = ARIMA(train['Total_Sales'], order=order)
        fitted_model = model.fit()
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        print(f"ARIMA{order}: AIC={aic:.2f}, BIC={bic:.2f}")
        arima_results.append({
            'order': order,
            'aic': aic,
            'bic': bic,
            'model': fitted_model
        })
        
        if aic < best_aic:
            best_aic = aic
            best_arima_order = order
            best_arima_model = fitted_model
    except:
        print(f"ARIMA{order}: Failed to converge")

print(f"\nBest ARIMA Model: ARIMA{best_arima_order} (AIC={best_aic:.2f})")

# ============================================================================
# SARIMA MODELS FOR TOTAL SALES
# ============================================================================
print("\n" + "="*80)
print("SARIMA MODELS FOR TOTAL SALES")
print("="*80)

# Test SARIMA configurations with seasonal component
sarima_configs = [
    ((1, 1, 1), (1, 1, 1, 12)),
    ((2, 1, 1), (1, 1, 1, 12)),
    ((1, 1, 2), (1, 1, 1, 12)),
    ((1, 1, 1), (2, 1, 1, 12)),
    ((1, 1, 1), (1, 1, 2, 12))
]

best_sarima_aic = np.inf
best_sarima_order = None
sarima_results = []

print("\nTesting SARIMA configurations:")
print("-" * 60)

for order, seasonal_order in sarima_configs:
    try:
        model = SARIMAX(train['Total_Sales'], order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        print(f"SARIMA{order}x{seasonal_order}: AIC={aic:.2f}, BIC={bic:.2f}")
        sarima_results.append({
            'order': order,
            'seasonal_order': seasonal_order,
            'aic': aic,
            'bic': bic,
            'model': fitted_model
        })
        
        if aic < best_sarima_aic:
            best_sarima_aic = aic
            best_sarima_order = (order, seasonal_order)
            best_sarima_model = fitted_model
    except Exception as e:
        print(f"SARIMA{order}x{seasonal_order}: Failed - {str(e)[:50]}")

print(f"\nBest SARIMA Model: SARIMA{best_sarima_order[0]}x{best_sarima_order[1]} (AIC={best_sarima_aic:.2f})")

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("MODEL EVALUATION ON TEST SET")
print("="*80)

# ARIMA Predictions
arima_forecast = best_arima_model.forecast(steps=len(test))
arima_rmse = np.sqrt(mean_squared_error(test['Total_Sales'], arima_forecast))
arima_mae = mean_absolute_error(test['Total_Sales'], arima_forecast)
arima_mape = mean_absolute_percentage_error(test['Total_Sales'], arima_forecast) * 100

print(f"\nARIMA{best_arima_order} Performance:")
print(f"  RMSE: {arima_rmse:.4f}")
print(f"  MAE: {arima_mae:.4f}")
print(f"  MAPE: {arima_mape:.2f}%")

# SARIMA Predictions
sarima_forecast = best_sarima_model.forecast(steps=len(test))
sarima_rmse = np.sqrt(mean_squared_error(test['Total_Sales'], sarima_forecast))
sarima_mae = mean_absolute_error(test['Total_Sales'], sarima_forecast)
sarima_mape = mean_absolute_percentage_error(test['Total_Sales'], sarima_forecast) * 100

print(f"\nSARIMA{best_sarima_order[0]}x{best_sarima_order[1]} Performance:")
print(f"  RMSE: {sarima_rmse:.4f}")
print(f"  MAE: {sarima_mae:.4f}")
print(f"  MAPE: {sarima_mape:.2f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# ARIMA Forecast
axes[0].plot(train.index, train['Total_Sales'], label='Training Data', color='blue', linewidth=2)
axes[0].plot(test.index, test['Total_Sales'], label='Actual Test Data', color='green', linewidth=2)
axes[0].plot(test.index, arima_forecast, label=f'ARIMA{best_arima_order} Forecast', 
             color='red', linewidth=2, linestyle='--')
axes[0].set_title(f'ARIMA{best_arima_order} Forecast vs Actual (RMSE: {arima_rmse:.4f})', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Total Sales (Millions)', fontsize=12)
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# SARIMA Forecast
axes[1].plot(train.index, train['Total_Sales'], label='Training Data', color='blue', linewidth=2)
axes[1].plot(test.index, test['Total_Sales'], label='Actual Test Data', color='green', linewidth=2)
axes[1].plot(test.index, sarima_forecast, label=f'SARIMA Forecast', 
             color='orange', linewidth=2, linestyle='--')
axes[1].set_title(f'SARIMA{best_sarima_order[0]}x{best_sarima_order[1]} Forecast vs Actual (RMSE: {sarima_rmse:.4f})', 
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('Total Sales (Millions)', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(script_dir, 'arima_sarima_forecasts.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\n✓ Forecast plots saved as 'arima_sarima_forecasts.png'")

# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')

# ARIMA Residuals
arima_residuals = test['Total_Sales'] - arima_forecast
axes[0, 0].plot(test.index, arima_residuals, color='red', linewidth=1.5)
axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0, 0].set_title(f'ARIMA{best_arima_order} Residuals', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Residuals', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(arima_residuals, bins=20, color='red', alpha=0.7, edgecolor='black')
axes[0, 1].set_title(f'ARIMA{best_arima_order} Residual Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Residuals', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# SARIMA Residuals
sarima_residuals = test['Total_Sales'] - sarima_forecast
axes[1, 0].plot(test.index, sarima_residuals, color='orange', linewidth=1.5)
axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1, 0].set_title(f'SARIMA Residuals', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Residuals', fontsize=10)
axes[1, 0].set_xlabel('Date', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(sarima_residuals, bins=20, color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].set_title(f'SARIMA Residual Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Residuals', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(script_dir, 'residual_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("✓ Residual analysis plots saved as 'residual_analysis.png'")

print("\n" + "="*80)
print("MODELING COMPLETE!")
print("="*80)
