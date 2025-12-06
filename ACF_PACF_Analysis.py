"""
ACF and PACF Analysis for Model Identification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ACF AND PACF ANALYSIS")
print("="*80)

# Load data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'VehicleData-1.csv')
df = pd.read_csv(data_path)
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')
df.columns = ['Total_Sales', 'New_Orders']

# Calculate differences
df['Sales_Diff'] = df['Total_Sales'].diff()
df['Orders_Diff'] = df['New_Orders'].diff()

# Create ACF/PACF plots
fig, axes = plt.subplots(4, 2, figsize=(16, 16))
fig.suptitle('ACF and PACF Analysis', fontsize=16, fontweight='bold')

# Total Sales - Original
plot_acf(df['Total_Sales'].dropna(), lags=40, ax=axes[0, 0], title='ACF: Total Sales (Original)')
plot_pacf(df['Total_Sales'].dropna(), lags=40, ax=axes[0, 1], title='PACF: Total Sales (Original)')

# Total Sales - Differenced
plot_acf(df['Sales_Diff'].dropna(), lags=40, ax=axes[1, 0], title='ACF: Total Sales (Differenced)')
plot_pacf(df['Sales_Diff'].dropna(), lags=40, ax=axes[1, 1], title='PACF: Total Sales (Differenced)')

# New Orders - Original
plot_acf(df['New_Orders'].dropna(), lags=40, ax=axes[2, 0], title='ACF: New Orders (Original)')
plot_pacf(df['New_Orders'].dropna(), lags=40, ax=axes[2, 1], title='PACF: New Orders (Original)')

# New Orders - Differenced
plot_acf(df['Orders_Diff'].dropna(), lags=40, ax=axes[3, 0], title='ACF: New Orders (Differenced)')
plot_pacf(df['Orders_Diff'].dropna(), lags=40, ax=axes[3, 1], title='PACF: New Orders (Differenced)')

plt.tight_layout()
output_path = os.path.join(script_dir, 'acf_pacf_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("\n✓ ACF/PACF analysis plot saved as 'acf_pacf_analysis.png'")

# Calculate and display ACF values
print("\nACF Values (First 12 lags):")
print("-" * 60)
acf_sales = acf(df['Sales_Diff'].dropna(), nlags=12)
acf_orders = acf(df['Orders_Diff'].dropna(), nlags=12)

print("\nTotal Sales (Differenced):")
for i, val in enumerate(acf_sales):
    print(f"  Lag {i}: {val:.4f}")

print("\nNew Orders (Differenced):")
for i, val in enumerate(acf_orders):
    print(f"  Lag {i}: {val:.4f}")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
