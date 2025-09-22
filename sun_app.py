# Sun Activity Predictor
# This implementation incorporates selected components and ideas from the open-source project:
# https://github.com/glassb/solar-forecast
# The referenced work provided valuable insights into solar forecasting techniques and data handling.

import tkinter as tk
from tkinter import filedialog, messagebox
# Tkinter GUI
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import warnings
from datetime import datetime, timedelta
from scipy.stats import sem
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from pvlib import location, irradiance
from datetime import datetime
import pytz

warnings.filterwarnings("ignore")
print("Modules Imported")

df = None
train_df = None
test_df = None

def select():
    global df
    try:

        # Select CSV file
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return  # User cancelled

        # Read CSV
        df = pd.read_csv(file_path)
        print("Test Csv Loaded")
        print(df.head(50))
            
        messagebox.showinfo(
            "Csv Read Complete"         
        )
            
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid test size between 0 and 1.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process CSV:\n{e}")

    return df
    
def prepare():
    global df
    if df is None:
        messagebox.showerror("No Data", "Please load a CSV file first.")
        return

    try:
        # Ensure datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by time to preserve sequence
        df = df.sort_values(by='Date').reset_index(drop=True)

        # Create target column: next time step's energy production
        df['Next-Energy-Value(Wh)'] = df['EnergyProd-Wh'].shift(-1)

        # Create lag features
        df['Previous-Energy-Value(Wh)'] = df['EnergyProd-Wh'].shift(1)
        df['Energy_Difference'] = df['EnergyProd-Wh'] - df['Previous-Energy-Value(Wh)']

        # Drop rows with NaNs from shifting
        df.dropna(inplace=True)

        # Compute daily energy production sums
        daily_sums = df.groupby('Date')['EnergyProd-Wh'].sum().reset_index()
        daily_sums.rename(columns={'EnergyProd-Wh': 'EnergyProd-Wh-Daily-Sum'}, inplace=True)
        df = df.merge(daily_sums, on='Date', how='left')

        # Define Cloud Index classification
        def classify_cloud(row):
            if row['EnergyProd-Wh-Daily-Sum'] == 0:
                return 'Dark'
            elif row['EnergyProd-Wh-Daily-Sum'] >= 4900:
                return 'Sunny'
            elif row['EnergyProd-Wh-Daily-Sum'] >= 3700:
                return 'Cloudy'
            elif row['EnergyProd-Wh-Daily-Sum'] > 0:
                return 'VeryCloudy'
            else:
                return ''

        df['Cloud_Index'] = df.apply(classify_cloud, axis=1)
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['EnergyProd-Wh-Daily-Sum'], label='Actual', color='blue')

        plt.xlabel('Date')
        plt.ylabel('Energy Production (Wh)')
        plt.title('Actual Energy Production')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        # Shuffle before splitting to avoid temporal bias (optional for non-time-series)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train and test
        split_ratio = 0.7
        split_index = int(len(df) * split_ratio)
        train_df = df[:split_index]
        test_df = df[split_index:]
            # 💾 Export to CSV
        train_df.to_csv('train.csv', index=False)
        test_df.to_csv('test.csv', index=False)
            # Print sample rows
        print(train_df.head(5))
        print(test_df.head(5))
            # Plot
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to prepare data:\n{e}")
        
def diag_s():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # 1. Cloud Index distribution
    print("Train Cloud Index distribution:")
    print(train_df['Cloud_Index'].value_counts())
    print("\nTest Cloud Index distribution:")
    print(test_df['Cloud_Index'].value_counts())

    # 2. Feature-target correlations
    print("\nFeature correlations with target:")
    corrs = train_df.corr(numeric_only=True)['Next-Energy-Value(Wh)'].sort_values(ascending=False)
    print(corrs)

    # 3. Sanity check with shuffled target
    shuffled_target = train_df['Next-Energy-Value(Wh)'].sample(frac=1, random_state=42).reset_index(drop=True)
    X = train_df[["Month", "Solar_w/m2", "Temperature_F", "UV", "EnergyProd-Wh", "Energy_Difference"]]

    model = LinearRegression()
    model.fit(X, shuffled_target)
    y_pred = model.predict(X)

    r2 = r2_score(shuffled_target, y_pred)
    mae = mean_absolute_error(shuffled_target, y_pred)

    print("\nSanity check with shuffled target:")
    print("R²:", r2)
    print("MAE:", mae)

    # 3b. Plot shuffled target vs predictions
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=shuffled_target, y=y_pred, alpha=0.5)
    plt.xlabel("Shuffled Actual Energy Value (Wh)")
    plt.ylabel("Predicted Energy Value (Wh)")
    plt.title(f"Sanity Check (Shuffled Target) — R²={r2:.4f}, MAE={mae:.2f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4. Visualize predictions vs actual for Sunny (real model)
    sunny_test = test_df[test_df['Cloud_Index'] == 'Sunny'].copy()
    sunny_test['Time_Minutes'] = pd.to_datetime(sunny_test['Time'], format='%H:%M:%S').dt.hour * 60 + \
                                 pd.to_datetime(sunny_test['Time'], format='%H:%M:%S').dt.minute

    X_sunny = sunny_test[["Month", "Time_Minutes", "Solar_w/m2", "Temperature_F", "UV", "EnergyProd-Wh", "Energy_Difference"]]
    y_sunny = sunny_test["Next-Energy-Value(Wh)"]

    model = LinearRegression()
    model.fit(X_sunny, y_sunny)
    y_pred_sunny = model.predict(X_sunny)

    #plt.figure(figsize=(8, 6))
    #sns.scatterplot(x=y_sunny, y=y_pred_sunny, alpha=0.5)
    #plt.xlabel("Actual Energy Value (Wh)")
    #plt.ylabel("Predicted Energy Value (Wh)")
    #plt.title("Sunny Model: Actual vs Predicted")
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()
def test():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # Ensure Time_Minutes is computed for both train and test
    for df in [train_df, test_df]:
        df['Time_Minutes'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour * 60 + \
                             pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute

    # Features for training and testing
    features = [
        "Month", "Time_Minutes",
        "Solar_w/m2", "Temperature_F", "UV",
        "EnergyProd-Wh", "Energy_Difference"
    ]

    # Train on full training set
    X_train = train_df[features]
    y_train = train_df["Next-Energy-Value(Wh)"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate per condition
    results = []
    for condition in ['Sunny', 'Cloudy', 'VeryCloudy']:
        subset = test_df[test_df['Cloud_Index'] == condition]
        X_test = subset[features]
        y_test = subset["Next-Energy-Value(Wh)"]

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results.append((condition, round(r2, 3), round(mae, 2)))

    # Print results table
    results_df = pd.DataFrame(results, columns=["Condition", "R²", "MAE"])
    print("\n📊 Linear Regression Performance by Condition:")
    print(results_df.to_string(index=False))

def test_with_plots():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # Ensure Time_Minutes is computed
    for df in [train_df, test_df]:
        df['Time_Minutes'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour * 60 + \
                             pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute

    features = [
        "Month", "Time_Minutes",
        "Solar_w/m2", "Temperature_F", "UV",
        "EnergyProd-Wh", "Energy_Difference"
    ]

    # 1. Cloud Index distribution plot
    cloud_counts = test_df['Cloud_Index'].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=cloud_counts.index, y=cloud_counts.values, palette=['gold', 'skyblue', 'gray' , 'black'])
    plt.title("Test Set Cloud Index Distribution")
    plt.ylabel("Sample Count")
    plt.xlabel("Cloud Index")
    plt.tight_layout()
    plt.show()

    # 2. Train model on full training set
    X_train = train_df[features]
    y_train = train_df["Next-Energy-Value(Wh)"]
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3. Evaluate and plot per condition
    results = []
    for condition in ['Sunny', 'Cloudy', 'VeryCloudy']:
        subset = test_df[test_df['Cloud_Index'] == condition]
        X_test = subset[features]
        y_test = subset["Next-Energy-Value(Wh)"]
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append((condition, round(r2, 3), round(mae, 2)))
        # Scatter plot with diagonal
        plt.figure(figsize=(7, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.xlabel("Actual Energy Value (Wh)")
        plt.ylabel("Predicted Energy Value (Wh)")
        plt.title(f"{condition} — Actual vs Predicted\nR²={r2:.3f}, MAE={mae:.2f} Wh")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
                # Compute residuals
        residuals = y_test - y_pred

        # Residual histogram
        plt.figure(figsize=(7, 5))
        sns.histplot(residuals, bins=40, kde=True, color='steelblue')
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        plt.title(f"{condition} — Residual Distribution\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}")
        plt.xlabel("Residual (Actual - Predicted) [Wh]")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Print results table
    results_df = pd.DataFrame(results, columns=["Condition", "R²", "MAE"])
    print("\n📊 Linear Regression Performance by Condition:")
    print(results_df.to_string(index=False))

def prepare_rf():
    global df, train_df, test_df

    # Ensure datetime format and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'Time']).reset_index(drop=True)

    # -----------------------------
    # 1️⃣ Create target (future value)
    # -----------------------------
    # Target is the NEXT timestep's energy production
    df['Next-Energy-Value(Wh)'] = df['EnergyProd-Wh'].shift(-1)

    # -----------------------------
    # 2️⃣ Create lag features (past only)
    # -----------------------------
    # These are safe because they only use data from the previous timestep
    df['Prev-Energy-Value(Wh)'] = df['EnergyProd-Wh'].shift(1)
    df['Prev-Solar_w/m2'] = df['Solar_w/m2'].shift(1)
    df['Prev-Temperature_F'] = df['Temperature_F'].shift(1)
    df['Prev-UV'] = df['UV'].shift(1)

    # Energy difference from previous timestep
    df['Energy_Difference'] = df['Prev-Energy-Value(Wh)'].diff()

    # -----------------------------
    # 3️⃣ Remove rows with NaNs from shifting
    # -----------------------------
    df.dropna(inplace=True)

    # -----------------------------
    # 4️⃣ Compute daily aggregates (safe)
    # -----------------------------
    daily_sums = df.groupby('Date')['EnergyProd-Wh'].sum().reset_index()
    daily_sums.rename(columns={'EnergyProd-Wh': 'EnergyProd-Wh-Daily-Sum'}, inplace=True)
    df = df.merge(daily_sums, on='Date', how='left')

    # -----------------------------
    # 5️⃣ Cloud Index classification
    # -----------------------------
    def classify_cloud(row):
        if row['EnergyProd-Wh-Daily-Sum'] == 0:
            return 'Dark'
        elif row['EnergyProd-Wh-Daily-Sum'] >= 4900:
            return 'Sunny'
        elif row['EnergyProd-Wh-Daily-Sum'] >= 3700:
            return 'Cloudy'
        elif row['EnergyProd-Wh-Daily-Sum'] > 0:
            return 'VeryCloudy'
        else:
            return 'Unknown'

    df['Cloud_Index'] = df.apply(classify_cloud, axis=1)

    # -----------------------------
    # 6️⃣ Optional: time in minutes
    # -----------------------------
    df['Time_Minutes'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour * 60 + \
                         pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute

    # -----------------------------
    # 7️⃣ Train-test split
    # -----------------------------
    # For time series, keep chronological order (no shuffle)
    split_ratio = 0.7
    split_index = int(len(df) * split_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print("✅ Data prepared without leakage")
    print(train_df.head(3))
    print(test_df.head(3))
    
def diag_rf():

    global train_df, test_df

    # 1. Cloud Index distribution
    print("📊 Train Cloud Index distribution:")
    print(train_df['Cloud_Index'].value_counts())
    print("\n📊 Test Cloud Index distribution:")
    print(test_df['Cloud_Index'].value_counts())

    # 2. Feature-target correlations
    print("\n🔍 Feature correlations with target:")
    corrs = train_df.corr(numeric_only=True)['Next-Energy-Value(Wh)'].sort_values(ascending=False)
    print(corrs)

    # 3. Sanity check with shuffled target
    print("\n🧪 Sanity check with shuffled target (Random Forest):")
    features = ["Month", "Prev-Solar_w/m2", "Prev-Temperature_F", "Prev-UV",
                "Prev-Energy-Value(Wh)", "Energy_Difference"]
    X = train_df[features].reset_index(drop=True)
    y_shuffled = train_df['Next-Energy-Value(Wh)'].sample(frac=1, random_state=42).reset_index(drop=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y_shuffled, test_size=0.3, random_state=42)

    rf_sanity = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    rf_sanity.fit(X_train, y_train)
    y_pred_val = rf_sanity.predict(X_val)

    r2_sanity = r2_score(y_val, y_pred_val)
    mae_sanity = mean_absolute_error(y_val, y_pred_val)
    print(f"R²: {r2_sanity:.6f}")
    print(f"MAE: {mae_sanity:.2f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_val, y=y_pred_val, alpha=0.5)
    plt.xlabel("Shuffled Actual Energy Value (Wh)")
    plt.ylabel("Predicted Energy Value (Wh)")
    plt.title(f"Sanity Check (RF): R²={r2_sanity:.3f}, MAE={mae_sanity:.2f}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def test_rf_with_plots():

    # 1. Load train/test data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # 2. Compute Time_Minutes for both datasets
    for df in [train_df, test_df]:
        df['Time_Minutes'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour * 60 + \
                             pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute

    # 3. Define features
    features = [
        "Month", "Time_Minutes",
        "Solar_w/m2", "Temperature_F", "UV",
        "EnergyProd-Wh", "Energy_Difference"
    ]

    # 4. Plot Cloud Index distribution
    cloud_counts = test_df['Cloud_Index'].value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=cloud_counts.index, y=cloud_counts.values, palette=['gold', 'skyblue', 'gray', 'black'])
    plt.title("Test Set Cloud Index Distribution")
    plt.ylabel("Sample Count")
    plt.xlabel("Cloud Index")
    plt.tight_layout()
    plt.show()

    # 5. Train RF model on full training set
    X_train = train_df[features]
    y_train = train_df["Next-Energy-Value(Wh)"]
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 6. Evaluate and plot per condition
    results = []
    for condition in ['Sunny', 'Cloudy', 'VeryCloudy']:
        subset = test_df[test_df['Cloud_Index'] == condition]
        X_test = subset[features]
        y_test = subset["Next-Energy-Value(Wh)"]
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results.append((condition, round(r2, 3), round(mae, 2)))

        # Scatter plot with diagonal
        plt.figure(figsize=(7, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        plt.xlabel("Actual Energy Value (Wh)")
        plt.ylabel("Predicted Energy Value (Wh)")
        plt.title(f"{condition} — Actual vs Predicted\nR²={r2:.3f}, MAE={mae:.2f} Wh")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Residual histogram
        residuals = y_test - y_pred
        plt.figure(figsize=(7, 5))
        sns.histplot(residuals, bins=40, kde=True, color='steelblue')
        plt.axvline(0, color='red', linestyle='--', label='Zero Error')
        plt.title(f"{condition} — Residual Distribution\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}")
        plt.xlabel("Residual (Actual - Predicted) [Wh]")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 7. Print results table
    results_df = pd.DataFrame(results, columns=["Condition", "R²", "MAE"])
    print("\n📊 Random Forest Performance by Condition:")
    print(results_df.to_string(index=False))

        
root = tk.Tk()
root.title("Sun Forecast Program")
root.geometry("660x200")

# Frame for top row of buttons
top_frame = tk.Frame(root)
top_frame.pack(pady=35)

btn_select = tk.Button(top_frame, text="Select CSV", command=select)
btn_select.pack(side="left", padx=5)

btn_prepare = tk.Button(top_frame, text="Prepare", command=prepare)
btn_prepare.pack(side="left", padx=5)

btn_diag = tk.Button(top_frame, text="Diagnose", command=diag_s)
btn_diag.pack(side="left", padx=5)

btn_test = tk.Button(top_frame, text="Test", command=test_with_plots)
btn_test.pack(side="left", padx=5)

# Frame for top row of buttons
bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=35)

btn_prepare_rf = tk.Button(bottom_frame, text="Prepare RF", command=prepare_rf)
btn_prepare_rf.pack(side="left", padx=5)

btn_diag_rf = tk.Button(bottom_frame, text="Diagnose RF", command=diag_rf)
btn_diag_rf.pack(side="left", padx=5)

btn_test_rf = tk.Button(bottom_frame, text="Test RF", command=test_rf_with_plots)
btn_test_rf.pack(side="left", padx=5)

root.mainloop()