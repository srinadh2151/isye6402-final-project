"""
Master Script to Run All Analyses
Executes all analysis scripts in sequence and organizes outputs
"""

import os
import sys
import subprocess
from datetime import datetime

print("="*80)
print("PASSENGER VEHICLE SUPPLY & DEMAND ANALYSIS")
print("Master Script - Running All Analyses")
print("="*80)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "="*80)

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(script_dir, 'outputs')

# Create outputs directory if it doesn't exist
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
    print("✓ Created 'outputs' directory")

# List of scripts to run in order
scripts = [
    ('Vehicle_Analysis_Complete.py', 'Exploratory Data Analysis & Stationarity Tests'),
    ('ACF_PACF_Analysis.py', 'ACF/PACF Analysis'),
    ('ARIMA_SARIMA_Modeling.py', 'ARIMA & SARIMA Modeling'),
    ('VAR_Analysis.py', 'Vector Autoregression Analysis'),
    ('ML_GradientBoosting_Analysis.py', 'Machine Learning: Gradient Boosting Methods')
]

# Track execution status
results = []

for i, (script, description) in enumerate(scripts, 1):
    print(f"\n{'='*80}")
    print(f"Step {i}/{len(scripts)}: {description}")
    print(f"Script: {script}")
    print(f"{'='*80}\n")
    
    try:
        # Run the script from its directory
        script_path = os.path.join(script_dir, script)
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(f"✓ {script} completed successfully")
            results.append((script, 'SUCCESS', None))
        else:
            print(f"✗ {script} failed with return code {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            results.append((script, 'FAILED', result.stderr))
    
    except subprocess.TimeoutExpired:
        print(f"✗ {script} timed out after 5 minutes")
        results.append((script, 'TIMEOUT', 'Script exceeded 5 minute timeout'))
    
    except Exception as e:
        print(f"✗ {script} encountered an error: {str(e)}")
        results.append((script, 'ERROR', str(e)))

# Move generated plots to outputs directory
print(f"\n{'='*80}")
print("Organizing Output Files")
print(f"{'='*80}\n")

plot_files = [
    'exploratory_analysis.png',
    'seasonal_decomposition.png',
    'acf_pacf_analysis.png',
    'arima_sarima_forecasts.png',
    'residual_analysis.png',
    'var_forecasts.png',
    'impulse_response.png',
    'fevd.png',
    'ml_gradient_boosting.png'
]

moved_count = 0
for plot_file in plot_files:
    source_path = os.path.join(script_dir, plot_file)
    dest_path = os.path.join(outputs_dir, plot_file)
    if os.path.exists(source_path):
        try:
            os.rename(source_path, dest_path)
            print(f"✓ Moved {plot_file} to outputs/")
            moved_count += 1
        except Exception as e:
            print(f"✗ Failed to move {plot_file}: {str(e)}")

print(f"\n✓ Moved {moved_count} plot files to outputs/ directory")

# Print summary
print(f"\n{'='*80}")
print("EXECUTION SUMMARY")
print(f"{'='*80}\n")

success_count = sum(1 for _, status, _ in results if status == 'SUCCESS')
failed_count = len(results) - success_count

print(f"Total Scripts: {len(results)}")
print(f"Successful: {success_count}")
print(f"Failed: {failed_count}")
print(f"\nDetailed Results:")
print("-" * 80)

for script, status, error in results:
    status_symbol = "✓" if status == 'SUCCESS' else "✗"
    print(f"{status_symbol} {script:40s} {status}")
    if error and status != 'SUCCESS':
        print(f"  Error: {error[:100]}...")

print(f"\n{'='*80}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")

if failed_count == 0:
    print("\n🎉 All analyses completed successfully!")
    print("\nGenerated outputs can be found in the 'outputs/' directory.")
    print("\nNext steps:")
    print("  1. Review the plots in the outputs/ directory")
    print("  2. Check the console output above for detailed results")
    print("  3. Refer to README.md for interpretation guidance")
else:
    print(f"\n⚠️  {failed_count} script(s) failed. Please review the errors above.")
    print("\nTroubleshooting:")
    print("  1. Ensure all required packages are installed: pip install -r requirements.txt")
    print("  2. Verify VehicleData-1.csv is in the current directory")
    print("  3. Check that you have write permissions in the current directory")

print("\n" + "="*80)
