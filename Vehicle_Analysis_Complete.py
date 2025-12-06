"""
Passenger Vehicle Supply & Demand Analysis
Complete Time Series Analysis Script
Author: Group 34
Date: November 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss, ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("PASSENGER VEHICLE SUPPLY & DEMAND ANALYSIS")
print("Time Series Analysis of U.S. Vehicle Sales and Manufacturing Orders")
print("Data Period: January 2000 - July 2024")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("1. DATA LOADING AND PREPROCESSING")
print("="*80)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'VehicleData-1.csv')
df = pd.read_csv(data_path)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')
df.columns = ['Total_Sales', 'New_Orders']

print(f"\nDataset Shape: {df.shape}")
print(f"Date Range: {df.index.min()} to {df.index.max()}")
print(f"Total Observations: {len(df)}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. EXPLORATORY DATA ANALYSIS")
print("="*80)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Time series plots
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df.index, df['Total_Sales'], linewidth=2, color='steelblue', label='Total Sales')
ax1.set_title('Total Vehicle Sales (Millions of Units, SAAR)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Sales (Millions)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--', alpha=0.5, label='Financial Crisis')
ax1.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', alpha=0.5, label='COVID-19')
ax1.legend()

ax2 = fig.add_subplot(gs[1, :])
ax2.plot(df.index, df['New_Orders'], linewidth=2, color='darkgreen', label='New Orders')
ax2.set_title('Manufacturers New Orders (Millions of Dollars, SA)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Orders (Millions $)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axvline(pd.Timestamp('2008-09-01'), color='red', linestyle='--', alpha=0.5)
ax2.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', alpha=0.5)

# Distribution plots
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(df['Total_Sales'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax3.set_title('Distribution of Total Sales', fontsize=12, fontweight='bold')
ax3.set_xlabel('Sales (Millions)')
ax3.set_ylabel('Frequency')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[2, 1])
ax4.hist(df['New_Orders'], bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
ax4.set_title('Distribution of New Orders', fontsize=12, fontweight='bold')
ax4.set_xlabel('Orders (Millions $)')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3)

# Scatter plot
ax5 = fig.add_subplot(gs[3, :])
ax5.scatter(df['New_Orders'], df['Total_Sales'], alpha=0.5, s=50)
ax5.set_title('Relationship: New Orders vs Total Sales', fontsize=12, fontweight='bold')
ax5.set_xlabel('New Orders (Millions $)')
ax5.set_ylabel('Total Sales (Millions)')
ax5.grid(True, alpha=0.3)

# Add correlation coefficient
corr = df['Total_Sales'].corr(df['New_Orders'])
ax5.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax5.transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

output_path = os.path.join(script_dir, 'exploratory_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\n✓ Exploratory analysis plot saved as 'exploratory_analysis.png'")

# Calculate correlation
print(f"\nCorrelation between Total Sales and New Orders: {corr:.4f}")

# ============================================================================
# 3. STATIONARITY TESTS
# ============================================================================
print("\n" + "="*80)
print("3. STATIONARITY ANALYSIS")
print("="*80)

def perform_stationarity_tests(series, name):
    """Perform ADF and KPSS tests for stationarity"""
    print(f"\n{name}:")
    print("-" * 60)
    
    # ADF Test
    adf_result = adfuller(series.dropna())
    print(f"ADF Test:")
    print(f"  Test Statistic: {adf_result[0]:.4f}")
    print(f"  p-value: {adf_result[1]:.4f}")
    print(f"  Critical Values: {adf_result[4]}")
    if adf_result[1] < 0.05:
        print(f"  → Series is STATIONARY (reject H0)")
    else:
        print(f"  → Series is NON-STATIONARY (fail to reject H0)")
    
    # KPSS Test
    kpss_result = kpss(series.dropna(), regression='ct')
    print(f"\nKPSS Test:")
    print(f"  Test Statistic: {kpss_result[0]:.4f}")
    print(f"  p-value: {kpss_result[1]:.4f}")
    print(f"  Critical Values: {kpss_result[3]}")
    if kpss_result[1] > 0.05:
        print(f"  → Series is STATIONARY (fail to reject H0)")
    else:
        print(f"  → Series is NON-STATIONARY (reject H0)")
    
    return adf_result, kpss_result

# Test original series
adf_sales, kpss_sales = perform_stationarity_tests(df['Total_Sales'], "Total Sales (Original)")
adf_orders, kpss_orders = perform_stationarity_tests(df['New_Orders'], "New Orders (Original)")

# Test differenced series
df['Sales_Diff'] = df['Total_Sales'].diff()
df['Orders_Diff'] = df['New_Orders'].diff()

adf_sales_diff, kpss_sales_diff = perform_stationarity_tests(df['Sales_Diff'], "Total Sales (First Difference)")
adf_orders_diff, kpss_orders_diff = perform_stationarity_tests(df['Orders_Diff'], "New Orders (First Difference)")

# ============================================================================
# 4. SEASONAL DECOMPOSITION
# ============================================================================
print("\n" + "="*80)
print("4. SEASONAL DECOMPOSITION")
print("="*80)

# Decompose both series
decomp_sales = seasonal_decompose(df['Total_Sales'], model='additive', period=12)
decomp_orders = seasonal_decompose(df['New_Orders'], model='additive', period=12)

# Plot decomposition
fig, axes = plt.subplots(4, 2, figsize=(18, 14))
fig.suptitle('Seasonal Decomposition Analysis', fontsize=16, fontweight='bold', y=0.995)

# Total Sales
axes[0, 0].plot(df.index, df['Total_Sales'], color='steelblue')
axes[0, 0].set_ylabel('Observed', fontsize=10)
axes[0, 0].set_title('Total Sales', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(decomp_sales.trend.index, decomp_sales.trend, color='orange')
axes[1, 0].set_ylabel('Trend', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

axes[2, 0].plot(decomp_sales.seasonal.index, decomp_sales.seasonal, color='green')
axes[2, 0].set_ylabel('Seasonal', fontsize=10)
axes[2, 0].grid(True, alpha=0.3)

axes[3, 0].plot(decomp_sales.resid.index, decomp_sales.resid, color='red')
axes[3, 0].set_ylabel('Residual', fontsize=10)
axes[3, 0].set_xlabel('Date', fontsize=10)
axes[3, 0].grid(True, alpha=0.3)

# New Orders
axes[0, 1].plot(df.index, df['New_Orders'], color='darkgreen')
axes[0, 1].set_ylabel('Observed', fontsize=10)
axes[0, 1].set_title('New Orders', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(decomp_orders.trend.index, decomp_orders.trend, color='orange')
axes[1, 1].set_ylabel('Trend', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

axes[2, 1].plot(decomp_orders.seasonal.index, decomp_orders.seasonal, color='green')
axes[2, 1].set_ylabel('Seasonal', fontsize=10)
axes[2, 1].grid(True, alpha=0.3)

axes[3, 1].plot(decomp_orders.resid.index, decomp_orders.resid, color='red')
axes[3, 1].set_ylabel('Residual', fontsize=10)
axes[3, 1].set_xlabel('Date', fontsize=10)
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(script_dir, 'seasonal_decomposition.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\n✓ Seasonal decomposition plot saved as 'seasonal_decomposition.png'")

# Calculate strength of seasonality
def seasonal_strength(decomposition):
    """Calculate strength of seasonality"""
    var_resid = np.var(decomposition.resid.dropna())
    var_seasonal_resid = np.var((decomposition.seasonal + decomposition.resid).dropna())
    return max(0, 1 - var_resid / var_seasonal_resid)

sales_seasonal_strength = seasonal_strength(decomp_sales)
orders_seasonal_strength = seasonal_strength(decomp_orders)

print(f"\nSeasonal Strength:")
print(f"  Total Sales: {sales_seasonal_strength:.4f}")
print(f"  New Orders: {orders_seasonal_strength:.4f}")

print("\nScript execution completed successfully!")
print("="*80)
