# Passenger Vehicle Supply & Demand Analysis
## Time Series Analysis of U.S. Vehicle Sales and Manufacturing Orders (2000-2024)

**Course:** ISyE 6402 - Time Series Analysis  
**Group:** 34  
**Team Members:** Srinadh Raja Nidadana, Park Hyung Min  
**Date:** 06 December 2025

---

## Project Overview

This project presents a comprehensive time series analysis of U.S. passenger vehicle sales and manufacturers' new orders from January 2000 to July 2024. The analysis combines traditional statistical methods (ARIMA, SARIMA, VAR) with modern machine learning techniques (XGBoost, LightGBM) to forecast vehicle sales and understand supply-demand dynamics.

### Research Questions

1. **Predictability**: What characteristics of the time series contribute most to the predictability of vehicle sales?

2. **Supply-Demand Relationship**: What is the relationship between vehicle supply and consumer demand trends?

3. **External Factors**: What external or exogenous factors help predict vehicle sales?

### Key Findings

- **Moderate correlation** (r = 0.47) between Total Sales and New Orders
- **Weak seasonality** (1.7-3.1% contribution) with 12-month cycles
- **Machine learning superiority**: 68-74% better RMSE than traditional methods
- **LightGBM best overall**: RMSE 0.79, R² 0.79, MAPE 3.8%
- **Bidirectional Granger causality** with Sales strongly influencing Orders
- **Major economic shocks** (2008 crisis, COVID-19) significantly impacted forecasting

---

## Dataset Description

**Source:** Federal Reserve Economic Data (FRED)  
**File:** `VehicleData-1.csv`  
**Period:** January 2000 - July 2024  
**Observations:** 295 monthly records  
**Train-Test Split:** 80% training (236 obs), 20% test (59 obs)

### Variables

1. **Total Sales (TOTALSA)**
   - Total vehicle sales in millions of units
   - Seasonally Adjusted Annual Rate (SAAR)
   - Source: https://fred.stlouisfed.org/series/TOTALSA

2. **New Orders (AMVPNO)**
   - Manufacturers' new orders for motor vehicles and parts
   - Millions of dollars, seasonally adjusted
   - Source: https://fred.stlouisfed.org/series/AMVPNO

### Descriptive Statistics

| Statistic | Total Sales (M units) | New Orders (M $) |
|-----------|----------------------|------------------|
| Count | 295 | 295 |
| Mean | 15.97 | 45,441 |
| Std Dev | 2.22 | 9,751 |
| Min | 8.83 | 18,900 |
| Median | 16.63 | 43,297 |
| Max | 22.05 | 64,455 |

---

## Project Structure

```
Final_Project_Solution/
│
├── VehicleData-1.csv                      # Raw data file
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── Final_Project_Report_Optimized.tex     # Final LaTeX report
├── ISyE_6402_Final_Project_Report_Group34.pdf     # Compiled PDF report
│
├── Vehicle_Analysis_Complete.py           # EDA, stationarity, decomposition
├── ACF_PACF_Analysis.py                   # ACF/PACF for model identification
├── ARIMA_SARIMA_Modeling.py               # ARIMA and SARIMA models
├── VAR_Analysis.py                        # VAR, Granger causality, IRF, FEVD
├── ML_GradientBoosting_Analysis.py        # XGBoost and LightGBM models
├── Multi_Horizon_Forecasting.py           # 1, 3, 6, 12-month forecasts
├── run_all_analyses.py                    # Master script
│
├── Vehicle_Data_Analysis.ipynb            # Jupyter notebook version
│
└── outputs/                               # Generated plots and results
    ├── exploratory_analysis.png
    ├── seasonal_decomposition.png
    ├── acf_pacf_analysis.png
    ├── arima_sarima_forecasts.png
    ├── residual_analysis.png
    ├── var_forecasts.png
    ├── impulse_response.png
    ├── fevd.png
    ├── ml_gradient_boosting.png
    ├── multi_horizon_forecasts.png
    └── multi_horizon_results.csv
```

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project folder**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - pandas >= 1.3.0
   - numpy >= 1.21.0
   - statsmodels >= 0.13.0
   - matplotlib >= 3.4.0
   - seaborn >= 0.11.0
   - scikit-learn >= 0.24.0
   - xgboost >= 1.5.0
   - lightgbm >= 3.3.0

3. **Verify installation:**
   ```bash
   python -c "import pandas, numpy, statsmodels, matplotlib, seaborn, sklearn, xgboost, lightgbm; print('All packages installed successfully!')"
   ```

---

## Running the Analysis

Execute specific analyses:

```bash
# Exploratory analysis and stationarity tests
python Vehicle_Analysis_Complete.py

# ACF/PACF analysis
python ACF_PACF_Analysis.py

# ARIMA and SARIMA modeling
python ARIMA_SARIMA_Modeling.py

# VAR analysis
python VAR_Analysis.py

# Machine learning models
python ML_GradientBoosting_Analysis.py

# Multi-horizon forecasting
python Multi_Horizon_Forecasting.py
```

---

## Analysis Components

### 1. Exploratory Data Analysis
- **Script:** `Vehicle_Analysis_Complete.py`
- **Outputs:** 
  - Time series plots with major economic events (2008 crisis, COVID-19)
  - Distribution histograms
  - Scatter plot showing correlation (r = 0.47)
  - Seasonal decomposition with strength metrics

**Key Finding:** Weak seasonality (1.7-3.1% contribution)

### 2. Stationarity Analysis
- **Tests:** ADF (Augmented Dickey-Fuller), KPSS
- **Results:**
  - Total Sales: Borderline stationarity (ADF p=0.046, KPSS p=0.01)
  - New Orders: Non-stationary (ADF p=0.675, KPSS p=0.01)
  - First differencing required (d=1)

### 3. Seasonal Decomposition
- **Method:** Additive decomposition
- **Period:** 12 months
- **Seasonal Strength:** 0.017 (Sales), 0.031 (Orders)
- **Interpretation:** Series predominantly driven by trend and irregular components

### 4. ACF/PACF Analysis
- **Script:** `ACF_PACF_Analysis.py`
- **Purpose:** Model identification for ARIMA/SARIMA
- **Lags Analyzed:** Up to 40 lags
- **Output:** Diagnostic plots for original and differenced series

### 5. ARIMA/SARIMA Modeling
- **Script:** `ARIMA_SARIMA_Modeling.py`
- **Best ARIMA:** ARIMA(3,1,1) - RMSE 2.97, MAE 2.46, MAPE 17.7%
- **Best SARIMA:** SARIMA(1,1,1)(2,1,1,12) - RMSE 2.61, MAE 2.05, MAPE 14.9%
- **Selection:** AIC/BIC criteria
- **Improvement:** SARIMA 12% better than ARIMA

### 6. Vector Autoregression (VAR)
- **Script:** `VAR_Analysis.py`
- **Optimal Lag:** 8 (AIC criterion)
- **Performance:** RMSE 2.46 (Sales), 7,387 (Orders)
- **Components:**
  - **Granger Causality:** Bidirectional, Sales → Orders stronger
  - **Impulse Response Functions (IRF):** Dynamic relationships
  - **FEVD:** Sales 98% self-driven, Orders 91% self-driven

### 7. Machine Learning Models
- **Script:** `ML_GradientBoosting_Analysis.py`
- **Features:** 29 engineered features (lags, rolling stats, time features, differences)
- **Models:**
  - **XGBoost:** RMSE 1.03, MAE 0.53, MAPE 4.1%, R² 0.64
  - **LightGBM:** RMSE 0.79, MAE 0.53, MAPE 3.8%, R² 0.79 ✅ **BEST**
- **Top Features:** Rolling max/mean/std, differences (1-month, 12-month)
- **Improvement:** 68-74% better than traditional methods

### 8. Multi-Horizon Forecasting
- **Script:** `Multi_Horizon_Forecasting.py`
- **Horizons:** 1, 3, 6, 12 months
- **Best by Horizon:**
  - **1-month:** XGBoost (RMSE 0.0003)
  - **3-month:** XGBoost (RMSE 0.09)
  - **6-month:** SARIMA (RMSE 0.30)
  - **12-month:** LightGBM (RMSE 1.38)

---

## Model Performance Comparison

### 1-Month Horizon (Primary Results)

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|-----|
| ARIMA(3,1,1) | 2.97 | 2.46 | 17.7% | -- |
| SARIMA(1,1,1)(2,1,1,12) | 2.61 | 2.05 | 14.9% | -- |
| VAR(8) | 2.46 | 1.89 | -- | -- |
| XGBoost | 1.03 | 0.53 | 4.1% | 0.64 |
| **LightGBM** | **0.79** | **0.53** | **3.8%** | **0.79** |

### Multi-Horizon Performance (RMSE)

| Model | 1-Month | 3-Month | 6-Month | 12-Month |
|-------|---------|---------|---------|----------|
| ARIMA | 0.10 | 0.45 | 0.62 | 3.84 |
| SARIMA | 0.34 | 0.29 | **0.30** | 3.56 |
| VAR | 0.09 | 0.30 | 0.37 | 3.54 |
| XGBoost | **0.00** | **0.09** | 2.53 | 2.05 |
| LightGBM | 0.15 | 0.10 | 1.33 | **1.38** |

---

## Key Findings and Insights

### 1. Model Selection by Use Case

**Short-Term Forecasting (1-3 months):**
- **Recommended:** XGBoost
- **Accuracy:** Near-perfect 1-month forecasts (RMSE 0.0003)
- **Use Case:** Operational planning, inventory management

**Medium-Term Forecasting (6 months):**
- **Recommended:** SARIMA
- **Accuracy:** RMSE 0.30, MAPE 1.58%
- **Use Case:** Quarterly planning, budget allocation

**Long-Term Forecasting (12 months):**
- **Recommended:** LightGBM
- **Accuracy:** RMSE 1.38, MAPE 7.94%
- **Use Case:** Annual forecasting, strategic planning

### 2. Supply-Demand Dynamics

- **Correlation:** Moderate (r = 0.47)
- **Causality:** Bidirectional with Sales → Orders stronger
- **Lead-Lag:** Orders provide moderate predictive value
- **Independence:** Both series 90%+ driven by own dynamics

### 3. Economic Shock Impact

**2008 Financial Crisis:**
- Drop: 44% (from 16.5M to 9.2M units)
- Recovery: Gradual over 5+ years

**COVID-19 Pandemic:**
- Drop: 49% (to 8.83M units, series minimum)
- Recovery: Rapid but volatile
- Impact: Test period (2019-2024) extremely challenging

### 4. Feature Importance

**Top Predictors (ML Models):**
1. Rolling statistics (max, mean, std) - 3, 6, 12-month windows
2. Recent differences (1-month, 12-month)
3. Lag features (1, 2, 3 months)
4. New Orders features (moderate contribution)

**Not Important:**
- Simple time trends
- Long lags (>12 months)
- Seasonal indicators (weak seasonality)

---

## Practical Recommendations

### For Automotive Industry

1. **Production Planning:**
   - Use XGBoost for 1-3 month operational forecasts
   - Monitor New Orders as leading indicator
   - Expect 60-70% accuracy improvement over traditional methods

2. **Inventory Management:**
   - SARIMA effective for 6-month planning
   - Account for weak seasonality (1.7-3.1% variation)
   - Focus on trend and irregular components

3. **Strategic Planning:**
   - LightGBM for 12-month forecasts
   - Incorporate rolling statistics and differences
   - Retrain models quarterly to capture market shifts

### For Policymakers

1. **Economic Indicators:**
   - Vehicle sales reflect consumer confidence
   - 1-2 month lag between orders and sales
   - Major shocks cause 40-50% drops

2. **Crisis Response:**
   - Traditional models fail during disruptions
   - ML models maintain 60-65% better accuracy
   - Consider regime-switching models for future work

---

## Limitations

### Data Limitations
1. Limited to 2 variables (missing gas prices, interest rates, unemployment)
2. Monthly aggregation may mask within-month dynamics
3. Structural breaks (2008, 2020) not explicitly modeled
4. National-level data obscures regional variation

### Methodological Limitations
1. Traditional methods assume linear relationships
2. ML models are "black boxes" with limited interpretability
3. Single train-test split limits validation robustness
4. Limited hyperparameter tuning
5. Multi-horizon forecasts use direct method (not iterative)

### Practical Limitations
1. Models require retraining for real-time forecasting
2. Point forecasts only (no prediction intervals for ML)
3. Requires future values of exogenous variables

---

## Future Work

### Data Enhancements
- Incorporate economic indicators (GDP, unemployment, consumer confidence)
- Add price variables (gas prices, interest rates)
- Include demographic factors and technology trends (EV adoption)
- Use higher frequency data (weekly/daily)
- Analyze disaggregated data (vehicle type, manufacturer, regional)

### Methodological Extensions
- Advanced time series models (regime-switching, GARCH, Prophet)
- Deep learning (LSTM, TCN, Transformers)
- Ensemble methods (combining SARIMA and XGBoost)
- Causal inference (structural VAR, instrumental variables)
- Probabilistic forecasting (quantile regression, Bayesian methods)

### Practical Applications
- Production planning and inventory optimization
- Dynamic pricing strategies
- Policy analysis (tax incentives, emission regulations)
- Risk management (stress testing, early warning systems)

---

## Technical Notes

### Software Versions
- Python: 3.8+
- pandas: 1.3.0+
- numpy: 1.21.0+
- statsmodels: 0.13.0+
- matplotlib: 3.4.0+
- seaborn: 0.11.0+
- scikit-learn: 0.24.0+
- xgboost: 1.5.0+
- lightgbm: 3.3.0+

### Reproducibility
- Random seed set (42) for all ML models
- All hyperparameters documented
- Train-test split strictly temporal
- All results validated and verified

## References

### Data Sources
1. Federal Reserve Bank of St. Louis. (2024). *FRED Economic Data*. https://fred.stlouisfed.org

### Methodology
1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

3. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.

4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.

5. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NIPS '17*.

---

## Citation

If you use this analysis or code, please cite:

```
Nidadana, S. R., & Park, H. M. (2025). Passenger Vehicle Supply & Demand Analysis: 
Time Series Analysis of U.S. Vehicle Sales and Manufacturing Orders (2000-2024). 
ISyE 6402 Final Project, Georgia Institute of Technology.
```

---

## License

This project is submitted as part of academic coursework at Georgia Institute of Technology. All data sources are publicly available from FRED (Federal Reserve Economic Data).

---

## Acknowledgments

- U.S. Bureau of Economic Analysis for data collection
- U.S. Census Bureau for manufacturing data
- Federal Reserve Bank of St. Louis for FRED database
- ISyE 6402 course instructors and TAs
- Georgia Institute of Technology

---

## Contact

For questions or issues:
- **Course:** ISyE 6402 - Time Series Analysis
- **Institution:** Georgia Institute of Technology
- **Semester:** Fall 2025

---

**Last Updated:** December 6, 2024  
**Version:** 2.0 (Final)