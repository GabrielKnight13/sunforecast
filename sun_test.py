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

print("Modules Imported")
df = None
global selected_reg_value
selected_reg_value = None
global selected_reg_value_model
selected_reg_value_model = None

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
        
def prepare():
    global df
   
    print(df.head())
    # Add Predictive energy Value and Previous Period's Energy value
    df['Next-Energy-Value(Wh)'] = np.nan
    df['Energy_Difference'] = np.nan
    df['Previous-Energy-Value(Wh)'] = np.nan

    for i in range(1, len(df)):
        prev_value = df.iloc[i - 1]['EnergyProd-Wh'] #iloc index with integer
        ftr_value = df.iloc[i]['EnergyProd-Wh']      #we can use loc with label
        slope = ftr_value - prev_value
        df.at[i, 'Next-Energy-Value(Wh)'] = ftr_value
        df.at[i, 'Energy_Difference'] = slope
        df.at[i,'Previous-Energy-Value(Wh)'] = prev_value
    #print(df.head(50))
    # Optional: fill first row manually if needed
    df.at[0, 'Previous-Energy-Value(Wh)'] = df.iloc[0]['EnergyProd-Wh']
    df.at[0, 'Next-Energy-Value(Wh)'] = df.iloc[1]['EnergyProd-Wh']
    df.at[0, 'Energy_Difference'] = df.iloc[1]['EnergyProd-Wh'] - df.iloc[0]['EnergyProd-Wh']
    # data cleaning commands
    df['Next-Energy-Value(Wh)'] = pd.to_numeric(df['Next-Energy-Value(Wh)'])
    df['Previous-Energy-Value(Wh)'] = pd.to_numeric(df['Previous-Energy-Value(Wh)'])
    
    df = df.dropna()
    #print(df_merged.head(100))
    df_merged_grouped = df.groupby('Date')
    
    #print(df_merged_grouped.head(10))
    #print(df_merged_grouped.describe())
    #print(df.head(10))
    #print(df.tail(10))
    # Plot Cloud Index distribution
    cloud_counts = df['Cloud_Index'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    cloud_counts.plot(kind='bar', color=['skyblue', 'gold', 'gray' , 'black'])
    plt.title('Frequency of Cloud Index Categories')
    plt.xlabel('Cloud Index')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['EnergyProd-Wh-Daily-Sum'], label='Actual', color='blue')

    plt.xlabel('Date')
    plt.ylabel('Energy Production (Wh)')
    plt.title('Actual Energy Production')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def model():
    global df, sunny_model, cloudy_model, verycloudy_model,selected_reg_value
   
    # SUNNY MODEL
    sunny_df = df[df['Cloud_Index'] == 'Sunny'].copy()
    sunny_df['Time_Minutes'] = pd.to_datetime(sunny_df['Time'], format='%H:%M:%S').dt.hour * 60 + \
                               pd.to_datetime(sunny_df['Time'], format='%H:%M:%S').dt.minute

    features = sunny_df[[
        "Month", "Time_Minutes", "Solar_w/m2", "Temperature_F", "UV",
        "EnergyProd-Wh", "Energy_Difference"
    ]]
    target = sunny_df["Next-Energy-Value(Wh)"]

    print("Features shape:", features.shape)
    print("Target shape:", target.shape)

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)
    # Create a StringVar to hold the selected value
      
    selected_value = selected_reg_value_model.get()

    if selected_value == "Linear Regression":
        print("LinearRegressor")
        sunny_model = LinearRegression()
        sunny_model.fit(features_train, target_train)
    elif selected_value == "RandomForestRegressor":
        print("RandomForestRegressor")
        sunny_model = RandomForestRegressor()
        sunny_model.fit(features_train, target_train)
    else:
        print("No valid selection")
    
    print('\n\nSUNNY MODEL')
    target_predict = sunny_model.predict(features_test)
    MSE = mean_squared_error(target_test, target_predict)
    R2 = r2_score(target_test, target_predict)
    
    SE = sem(target_test - target_predict)
  
    print('Standard Error:', SE)
    print("Mean Squared Error:", MSE)
    print("R-squared:", R2)
   
    train_predict = sunny_model.predict(features_train)
    mae_train = mean_absolute_error(target_train, train_predict)
    mae_test = mean_absolute_error(target_test, target_predict)

    print(f"MAE Train: {mae_train:.4f}")
    print(f"MAE Test: {mae_test:.4f}")
    # CLOUDY MODEL
    cloudy_df = df[df['Cloud_Index'] == 'Cloudy'].copy()
    cloudy_df['Time_Minutes'] = pd.to_datetime(cloudy_df['Time'], format='%H:%M:%S').dt.hour * 60 + \
                                pd.to_datetime(cloudy_df['Time'], format='%H:%M:%S').dt.minute

    features = cloudy_df[[
        "Month", "Time_Minutes", "Solar_w/m2", "Temperature_F", "UV",
        "EnergyProd-Wh", "Energy_Difference"
    ]]
    target = cloudy_df["Next-Energy-Value(Wh)"]

    print("Features shape:", features.shape)
    print("Target shape:", target.shape)

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)
    
    if selected_value == "Linear Regression":
        print("LinearRegressor")
        cloudy_model = LinearRegression()
        cloudy_model.fit(features_train, target_train)
    elif selected_value == "RandomForestRegressor":
        print("RandomForestRegressor")
        cloudy_model = RandomForestRegressor()
        cloudy_model.fit(features_train, target_train)
    else:
        print("No valid selection")
 
    print('\n\nCLOUDY MODEL')
    target_predict = cloudy_model.predict(features_test)
    MSE = mean_squared_error(target_test, target_predict)
    R2 = r2_score(target_test, target_predict)
       
    SE = sem(target_test - target_predict)

    print('Standard Error:', SE)
    print("Mean Squared Error:", MSE)
    print("R-squared:", R2)
    train_predict = cloudy_model.predict(features_train)
    mae_train = mean_absolute_error(target_train, train_predict)
    mae_test = mean_absolute_error(target_test, target_predict)

    print(f"MAE Train: {mae_train:.4f}")
    print(f"MAE Test: {mae_test:.4f}")

    # VERY CLOUDY MODEL
    verycloudy_df = df[df['Cloud_Index'] == 'VeryCloudy'].copy()
    verycloudy_df['Time_Minutes'] = pd.to_datetime(verycloudy_df['Time'], format='%H:%M:%S').dt.hour * 60 + \
                                    pd.to_datetime(verycloudy_df['Time'], format='%H:%M:%S').dt.minute
    features = verycloudy_df[[
        "Month", "Time_Minutes", "Solar_w/m2", "Temperature_F", "UV",
        "EnergyProd-Wh", "Energy_Difference"
    ]]
    target = verycloudy_df["Next-Energy-Value(Wh)"]

    print("Features shape:", features.shape)
    print("Target shape:", target.shape)

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)
    
    if selected_value == "Linear Regression":
        print("LinearRegressor")
        verycloudy_model = LinearRegression()
        verycloudy_model.fit(features_train, target_train)
    elif selected_value == "RandomForestRegressor":
        print("RandomForestRegressor")
        verycloudy_model = RandomForestRegressor()
        verycloudy_model.fit(features_train, target_train)
    else:
        print("No valid selection")

    print('\n\nVERY CLOUDY MODEL')
    target_predict = verycloudy_model.predict(features_test)
    MSE = mean_squared_error(target_test, target_predict)
    R2 = r2_score(target_test, target_predict)
    SE = np.sqrt(((target_test - target_predict) ** 2).sum() / (len(df) - 1))

    print('Standard Error:', SE)
    print("Mean Squared Error:", MSE)
    print("R-squared:", R2)
    train_predict = verycloudy_model.predict(features_train)
    mae_train = mean_absolute_error(target_train, train_predict)
    mae_test = mean_absolute_error(target_test, target_predict)

    print(f"MAE Train: {mae_train:.4f}")
    print(f"MAE Test: {mae_test:.4f}")

def test():
    global df, sunny_model, cloudy_model, verycloudy_model, df_prediction
    required_cols = [
        "Month", "Time", "Solar_w/m2", "Temperature_F", 
        "UV", "EnergyProd-Wh", "Energy_Difference", "Cloud_Index", "Next-Energy-Value(Wh)"
    ]
    start = 5
    end = 10000

    actual_values = []
    predicted_values = []
    months = []
    dates = []
    
    for i in range(start, end):
        df_temp = df.iloc[[i]].copy()
        df_temp = df_temp[required_cols]

        # Convert time to minutes
        df_temp['Time_Minutes'] = pd.to_datetime(df_temp['Time'], format='%H:%M:%S').dt.hour * 60 + \
                                  pd.to_datetime(df_temp['Time'], format='%H:%M:%S').dt.minute

        # Select features
        feature_cols = [
            "Month", "Time_Minutes", "Solar_w/m2", "Temperature_F", 
            "UV", "EnergyProd-Wh", "Energy_Difference"
        ]
        features = df_temp[feature_cols]
        cloud_index = df_temp.iloc[0]['Cloud_Index']

        # Choose model
        if cloud_index == 'Sunny':
            model = sunny_model
        elif cloud_index == 'Cloudy':
            model = cloudy_model
        elif cloud_index == 'VeryCloudy':
            model = verycloudy_model
        else:
            continue  # Skip unknown cloud types
        
        # Predict and store
        prediction = model.predict(features)[0]
        actual = df_temp.iloc[0]['Next-Energy-Value(Wh)']
        month = df_temp.iloc[0]['Month']
        date = pd.to_datetime(df.loc[i, "Date"])
        
        predicted_values.append(prediction)
        actual_values.append(actual)
        months.append(month)
        dates.append(date)
        
    # Ensure predicted_values is a 1D list of floats
    df_prediction = pd.DataFrame({
        "Predicted_Energy": predicted_values,
        "Actual_Energy": actual_values,
        "Month": months,
        "Date": dates
    })
    # Step 1: Calculate residuals
    df_prediction['Residual'] = df_prediction['Actual_Energy'] - df_prediction['Predicted_Energy']
    print(df_prediction.head(50))

        # Testing statistics
    print('\n\n')
    print('PREDICTION STATS')
    mae = (df_prediction['Actual_Energy'] - df_prediction['Predicted_Energy']).abs().mean()
    print('MAE:', mae)
    non_zero = df_prediction[df_prediction['Actual_Energy'] != 0]
    mape = ((non_zero['Actual_Energy'] - non_zero['Predicted_Energy']).abs() / non_zero['Actual_Energy']).mean() * 100
    print('MAPE (%):', mape)
    rmse = np.sqrt(((df_prediction['Actual_Energy'] - df_prediction['Predicted_Energy']) ** 2).mean())
    print('RMSE:', rmse)
def plot():
    global df, sunny_model, cloudy_model, verycloudy_model, df_prediction
    
    # Plot Actual and Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(df_prediction['Date'], df_prediction['Actual_Energy'], label='Actual', color='blue')
    plt.plot(df_prediction['Date'], df_prediction['Predicted_Energy'], label='Predicted', color='red')

    plt.xlabel('Date')
    plt.ylabel('Energy Production (Wh)')
    plt.title('Actual vs Predicted Energy Production')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
        #Residuals Over Time
    plt.figure(figsize=(12, 4))
    plt.plot(df_prediction['Date'], df_prediction['Residual'], label='Residuals', color='purple')
    plt.axhline(0, color='gray', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Residual (Actual − Predicted)')
    plt.title('Residuals Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    #Residual Distribution (Histogram)
    plt.figure(figsize=(8, 4))
    plt.hist(df_prediction['Residual'], bins=50, color='orange', edgecolor='black')

    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    plt.show()
def prepare_temporal_features(df):
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time_Minutes'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.hour * 60 + \
                         pd.to_datetime(df['Time'], format='%H:%M:%S').dt.minute

    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Cyclical encoding
    df['Time_sin'] = np.sin(2 * np.pi * df['Time_Minutes'] / 1440)
    df['Time_cos'] = np.cos(2 * np.pi * df['Time_Minutes'] / 1440)
    df['Day_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['Day_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

    return df[['Month', 'Time_sin', 'Time_cos', 'Day_sin', 'Day_cos']]
    
def forecast_solar(month, date_str, time_str, model):
    global selected_reg_value
    # Assume df contains historical data with Solar_w/m2
    X = prepare_temporal_features(df)
    y = df['Solar_w/m2']
    
    selected_value = selected_reg_value.get()

    if selected_value == "Linear Regression":
        print("LinearRegressor")
        solar_model = LinearRegression()
        solar_model.fit(X, y)
    elif selected_value == "RandomForestRegressor":
        print("RandomForestRegressor")
        solar_model = RandomForestRegressor()
        solar_model.fit(X, y)
    else:
        print("No valid selection")
    
    date = pd.to_datetime(date_str)
    time = pd.to_datetime(time_str, format='%H:%M:%S')

    time_minutes = time.hour * 60 + time.minute
    day_of_year = date.dayofyear

    input_data = pd.DataFrame([{
        'Month': month,
        'Time_sin': np.sin(2 * np.pi * time_minutes / 1440),
        'Time_cos': np.cos(2 * np.pi * time_minutes / 1440),
        'Day_sin': np.sin(2 * np.pi * day_of_year / 365),
        'Day_cos': np.cos(2 * np.pi * day_of_year / 365)
    }])

    prediction = solar_model.predict(input_data)[0]
    print(f"☀️ Forecasted Solar_w/m2 for {date_str} at {time_str}: {prediction:.2f}")
    return prediction
    
def forecast_day_solar(date_str, month, model):
    start_time = datetime.strptime("08:00:00", "%H:%M:%S")
    end_time = datetime.strptime("18:00:00", "%H:%M:%S")
    interval = timedelta(minutes=15)

    current_time = start_time
    results = []

    while current_time <= end_time:
        time_str = current_time.strftime("%H:%M:%S")
        prediction = forecast_solar(month=month, date_str=date_str, time_str=time_str, model=model)
        results.append({
            "Time": time_str,
            "Solar_w/m2": prediction
        })
        current_time += interval

    return results
    
def future_forecast():
    global entry, daily_forecast
    # One value estimation
    #pred = forecast_solar(month=10, date_str="2025-10-15", time_str="13:30:00", model=model)
    print("Daily estimation")
    date_str = entry.get()
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    month = date_obj.month
    daily_forecast = forecast_day_solar(date_str=date_str, month=month, model=model)

    #for entry in daily_forecast:
    #    print(f"{entry['Time']} → {entry['Solar_w/m2']:.2f} W/m²")
    total_solar = sum(entry['Solar_w/m2'] for entry in daily_forecast)

    print(f"\n🔆 Total Solar_w/m2 from 08:00 to 18:00: {total_solar:.2f}")
    total_solar = total_solar * 0.2 #Efficiency
    Cloud = None
    if total_solar == 0:
        Cloud = 'Dark'
    elif total_solar >= 4900:
        Cloud = 'Sunny'
    elif total_solar >= 3700:
        Cloud = 'Cloudy'
    elif total_solar > 0:
        Cloud = 'VeryCloudy'
    else:
        return ''
    print(Cloud)
    
def data_obtain():
    global df_actual
    # 📍 Define location for Istanbul
    latitude = 41.0082
    longitude = 28.9784
    tz = 'Europe/Istanbul'
    site = location.Location(latitude, longitude, tz=tz)
    date_str = entry.get()
    print(date_str)
    # 🕒 Define time range for September 13, 2024 to 25, from 08:00 to 18:00 local time
    start = pd.Timestamp(f"{date_str} 08:00:00", tz=tz)
    end = pd.Timestamp(f"{date_str} 18:00:00", tz=tz)
    times = pd.date_range(start=start, end=end, freq='15min')

    # ☀️ Compute clear-sky irradiance
    cs = site.get_clearsky(times)

    # 🌡️ Simulate temperature (°C) and UV index as if from a database
    weather_data = pd.DataFrame(index=times)
    weather_data['temperature_C'] = np.random.uniform(20, 30, size=len(times))
    weather_data['UV'] = np.random.uniform(3, 8, size=len(times))
    weather_data['temperature_F'] = weather_data['temperature_C'] * 9/5 + 32

    # ☀️ Compute solar position
    solar_position = site.get_solarposition(times)

    # 🧮 Calculate POA irradiance using clearsky model
    surface_tilt = 30
    surface_azimuth = 180

    poa_irradiance = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        dni=cs['dni'],
        ghi=cs['ghi'],
        dhi=cs['dhi']
    )

    # ⚡ Estimate energy production (Wh) for 1 m² panel with 20% efficiency
    efficiency = 0.2
    energy_production = poa_irradiance['poa_global'] * efficiency

    # 📊 Construct final DataFrame
    df_actual = pd.DataFrame({
        'Date': times.date,
        'Time': times.time,
        'Month': times.month,
        'Solar_w/m2': poa_irradiance['poa_global'].values,
        'Temperature_F': weather_data['temperature_C'].values,
        'UV': weather_data['UV'].values,
        'EnergyProd-Wh': energy_production.values
    })

    # Compute daily energy production sums
    df_actual['Date'] = pd.to_datetime(df['Date'])  # Ensure datetime format
    daily_sums = df_actual.groupby('Date')['EnergyProd-Wh'].sum().reset_index()
    daily_sums.rename(columns={'EnergyProd-Wh': 'EnergyProd-Wh-Daily-Sum'}, inplace=True)

    # Merge back into original DataFrame
    df_actual = df_actual.merge(daily_sums, on='Date', how='left')
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

        # Apply classification
    df_actual['Cloud_Index'] = df_actual.apply(classify_cloud, axis=1)

        # Optional: Preview
    print(df_actual['Cloud_Index'].value_counts())
    print(df_actual.head(100))
    print("Data obtained for date")
        # Plot Cloud Index distribution
    cloud_counts = df_actual['Cloud_Index'].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    cloud_counts.plot(kind='bar', color=['skyblue', 'gold', 'gray' , 'black'])
    plt.title('Frequency of Cloud Index Categories')
    plt.xlabel('Cloud Index')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return df_actual
def plot_est():   
    # Plot Actual and Predicted
    global daily_forecast,df_actual
    data_obtain()
    times = []
    solar_values = []
   
    for entry in daily_forecast:
        times.append(entry['Time'])
        solar_values.append(entry['Solar_w/m2'])
    
    df_actual['TimeStr'] = df_actual['Time'].astype(str)
    plt.figure(figsize=(12, 6))
    plt.plot(times, solar_values, label='Predicted', color='blue',marker='o')
    plt.plot(df_actual['TimeStr'], df_actual['Solar_w/m2'], label='Actual', color='red',marker='o')

    plt.xlabel('Time')
    plt.ylabel('Energy Production (Wh)')
    plt.title('Actual vs Predicted Energy Production')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("Estimated vs Actual Results")
    # Step 1: Calculate residuals
    actual_energy = df_actual['Solar_w/m2']
    residuals = actual_energy - solar_values
    print(residuals.head(50))
       #Residuals Over Time
    plt.figure(figsize=(12, 4))
    plt.plot(times, residuals, label='Residuals', color='purple')
    plt.axhline(0, color='gray', linestyle='--')

    plt.xlabel('Date')
    plt.ylabel('Residual (Actual − Predicted)')
    plt.title('Residuals Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    #Residual Distribution (Histogram)
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, color='orange', edgecolor='black')

    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    plt.show()
    
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

selected_reg_value_model = tk.StringVar()
    
# Create the ComboBox
combo = ttk.Combobox(top_frame, textvariable=selected_reg_value_model)
combo['values'] = ("Linear Regression", "RandomForestRegressor")  # List of choices
combo['state'] = 'readonly'  # Prevent typing custom values
combo.current(0)  # Set default selection
combo.pack(side="left", padx=15)
    
btn_model = tk.Button(top_frame, text="Model", command=model)
btn_model.pack(side="left", padx=5)

btn_test = tk.Button(top_frame, text="Test", command=test)
btn_test.pack(side="left", padx=5)

btn_plot = tk.Button(top_frame, text="Plot", command=plot)
btn_plot.pack(side="left", padx=5)

# Frame for date input and estimate button
bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=20)

lbl_date = tk.Label(bottom_frame, text="Enter Date: Y-M-D")
lbl_date.pack(side="left", padx=5)

entry = tk.Entry(bottom_frame, width=20)
entry.pack(side="left", padx=5)

btn_future = tk.Button(bottom_frame, text="Estimate", command=future_forecast)
btn_future.pack(side="left", padx=5)
selected_reg_value = tk.StringVar()
# Create the ComboBox
combo = ttk.Combobox(bottom_frame, textvariable=selected_reg_value)
combo['values'] = ("Linear Regression", "RandomForestRegressor")  # List of choices
combo['state'] = 'readonly'  # Prevent typing custom values
combo.current(0)  # Set default selection
combo.pack(side="left", padx=15)

btn_plot_est = tk.Button(bottom_frame, text="GetData&PlotResults", command=plot_est)
btn_plot_est.pack(side="left", padx=5)

root.mainloop()

