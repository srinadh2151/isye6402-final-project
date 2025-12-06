"""
Vector Autoregression (VAR) Analysis
Analyzing the relationship between Total Sales and New Orders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("VECTOR AUTOREGRESSION (VAR) ANALYSIS")
print("="*80)

# Load data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'VehicleData-1.csv')
df = pd.read_csv(data_path)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')
df.columns = ['Total_Sales', 'New_Orders']

# ============================================================================
# GRANGER CAUSALITY TEST
# ============================================================================
print("\n" + "="*80)
print("GRANGER CAUSALITY TESTS")
print("="*80)

# Test if New Orders Granger-causes Total Sales
print("\nTest: Does New Orders Granger-cause Total Sales?")
print("-" * 60)
try:
    gc_result_1 = grangercausalitytests(df[['Total_Sales', 'New_Orders']], maxlag=12, verbose=False)
    
    print("\nLag | F-statistic | p-value | Significant?")
    print("-" * 60)
    for lag in range(1, 13):
        f_stat = gc_result_1[lag][0]['ssr_ftest'][0]
        p_value = gc_result_1[lag][0]['ssr_ftest'][1]
        sig = "Yes" if p_value < 0.05 else "No"
        print(f"{lag:3d} | {f_stat:11.4f} | {p_value:7.4f} | {sig}")
except Exception as e:
    print(f"Error in Granger causality test: {e}")

# Test if Total Sales Granger-causes New Orders
print("\n\nTest: Does Total Sales Granger-cause New Orders?")
print("-" * 60)
try:
    gc_result_2 = grangercausalitytests(df[['New_Orders', 'Total_Sales']], maxlag=12, verbose=False)
    
    print("\nLag | F-statistic | p-value | Significant?")
    print("-" * 60)
    for lag in range(1, 13):
        f_stat = gc_result_2[lag][0]['ssr_ftest'][0]
        p_value = gc_result_2[lag][0]['ssr_ftest'][1]
        sig = "Yes" if p_value < 0.05 else "No"
        print(f"{lag:3d} | {f_stat:11.4f} | {p_value:7.4f} | {sig}")
except Exception as e:
    print(f"Error in Granger causality test: {e}")

# ============================================================================
# VAR MODEL FITTING
# ============================================================================
print("\n" + "="*80)
print("VAR MODEL FITTING")
print("="*80)

# Difference the data to make it stationary
df_diff = df.diff().dropna()

# Split data
train_size = int(len(df_diff) * 0.8)
train = df_diff[:train_size]
test = df_diff[train_size:]

print(f"\nData Split (Differenced):")
print(f"  Training set: {len(train)} observations")
print(f"  Test set: {len(test)} observations")

# Fit VAR model
print("\nFitting VAR model...")
model = VAR(train)

# Select optimal lag order
print("\nSelecting optimal lag order:")
print("-" * 60)
lag_order_results = model.select_order(maxlags=12)
print(lag_order_results.summary())

# Use AIC to select lag order
optimal_lag = lag_order_results.aic
print(f"\nOptimal lag order (AIC): {optimal_lag}")

# Fit VAR model with optimal lag
var_model = model.fit(optimal_lag)
print(f"\nVAR({optimal_lag}) Model Summary:")
print("=" * 60)
print(var_model.summary())

# ============================================================================
# FORECASTING
# ============================================================================
print("\n" + "="*80)
print("VAR MODEL FORECASTING")
print("="*80)

# Forecast
lag_order = var_model.k_ar
forecast_input = train.values[-lag_order:]
forecast = var_model.forecast(y=forecast_input, steps=len(test))

# Convert forecast to DataFrame
forecast_df = pd.DataFrame(forecast, index=test.index, columns=['Total_Sales_Diff', 'New_Orders_Diff'])

# Convert differences back to levels
# Reconstruct the levels from differences
last_train_values = df.iloc[train_size - 1]
forecast_levels = pd.DataFrame(index=test.index, columns=['Total_Sales', 'New_Orders'])

# Use .loc for proper assignment
forecast_levels.loc[forecast_levels.index[0], 'Total_Sales'] = last_train_values['Total_Sales'] + forecast_df.iloc[0]['Total_Sales_Diff']
forecast_levels.loc[forecast_levels.index[0], 'New_Orders'] = last_train_values['New_Orders'] + forecast_df.iloc[0]['New_Orders_Diff']

for i in range(1, len(forecast_levels)):
    forecast_levels.loc[forecast_levels.index[i], 'Total_Sales'] = forecast_levels.iloc[i-1]['Total_Sales'] + forecast_df.iloc[i]['Total_Sales_Diff']
    forecast_levels.loc[forecast_levels.index[i], 'New_Orders'] = forecast_levels.iloc[i-1]['New_Orders'] + forecast_df.iloc[i]['New_Orders_Diff']

# Calculate metrics
# Align the test data with forecast (account for differencing)
test_actual = df.iloc[train_size+1:]  # Skip first observation due to differencing
forecast_levels_aligned = forecast_levels.iloc[:len(test_actual)]

rmse_sales = np.sqrt(mean_squared_error(test_actual['Total_Sales'], forecast_levels_aligned['Total_Sales']))
mae_sales = mean_absolute_error(test_actual['Total_Sales'], forecast_levels_aligned['Total_Sales'])

rmse_orders = np.sqrt(mean_squared_error(test_actual['New_Orders'], forecast_levels_aligned['New_Orders']))
mae_orders = mean_absolute_error(test_actual['New_Orders'], forecast_levels_aligned['New_Orders'])

print(f"\nVAR Model Performance:")
print("-" * 60)
print(f"Total Sales:")
print(f"  RMSE: {rmse_sales:.4f}")
print(f"  MAE: {mae_sales:.4f}")
print(f"\nNew Orders:")
print(f"  RMSE: {rmse_orders:.4f}")
print(f"  MAE: {mae_orders:.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Total Sales
axes[0].plot(df.index[:train_size], df['Total_Sales'][:train_size], 
             label='Training Data', color='blue', linewidth=2)
axes[0].plot(test_actual.index, test_actual['Total_Sales'], 
             label='Actual Test Data', color='green', linewidth=2)
axes[0].plot(forecast_levels_aligned.index, forecast_levels_aligned['Total_Sales'], 
             label=f'VAR({optimal_lag}) Forecast', color='red', linewidth=2, linestyle='--')
axes[0].set_title(f'VAR({optimal_lag}) Forecast: Total Sales (RMSE: {rmse_sales:.4f})', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Total Sales (Millions)', fontsize=12)
axes[0].legend(loc='best', fontsize=10)
axes[0].grid(True, alpha=0.3)

# New Orders
axes[1].plot(df.index[:train_size], df['New_Orders'][:train_size], 
             label='Training Data', color='blue', linewidth=2)
axes[1].plot(test_actual.index, test_actual['New_Orders'], 
             label='Actual Test Data', color='green', linewidth=2)
axes[1].plot(forecast_levels_aligned.index, forecast_levels_aligned['New_Orders'], 
             label=f'VAR({optimal_lag}) Forecast', color='orange', linewidth=2, linestyle='--')
axes[1].set_title(f'VAR({optimal_lag}) Forecast: New Orders (RMSE: {rmse_orders:.4f})', 
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('New Orders (Millions $)', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(script_dir, 'var_forecasts.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\n✓ VAR forecast plots saved as 'var_forecasts.png'")

# ============================================================================
# IMPULSE RESPONSE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("IMPULSE RESPONSE ANALYSIS")
print("="*80)

irf = var_model.irf(10)

fig = irf.plot(orth=False, figsize=(16, 10))
plt.suptitle('Impulse Response Functions', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
output_path = os.path.join(script_dir, 'impulse_response.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\n✓ Impulse response plots saved as 'impulse_response.png'")

# ============================================================================
# FORECAST ERROR VARIANCE DECOMPOSITION
# ============================================================================
print("\n" + "="*80)
print("FORECAST ERROR VARIANCE DECOMPOSITION")
print("="*80)

fevd = var_model.fevd(10)
print("\n", fevd.summary())

fig = fevd.plot(figsize=(16, 8))
plt.suptitle('Forecast Error Variance Decomposition', fontsize=16, fontweight='bold')
plt.tight_layout()
output_path = os.path.join(script_dir, 'fevd.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\n✓ FEVD plots saved as 'fevd.png'")

print("\n" + "="*80)
print("VAR ANALYSIS COMPLETE!")
print("="*80)
