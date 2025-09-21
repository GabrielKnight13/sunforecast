import pandas as pd
import numpy as np
from pvlib import location, irradiance
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

# 📍 Define location for Istanbul
latitude = 41.0082
longitude = 28.9784
tz = 'Europe/Istanbul'
site = location.Location(latitude, longitude, tz=tz)

# 🕒 Define time range for September 13, 2024 to 25, from 08:00 to 18:00 local time
start = pd.Timestamp('2025-09-15 08:00:00', tz=tz)
end = pd.Timestamp('2025-09-15 18:00:00', tz=tz)
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
df = pd.DataFrame({
    'Date': times.date,
    'Time': times.time,
    'Month': times.month,
    'Solar_w/m2': poa_irradiance['poa_global'].values,
    'Temperature_F': weather_data['temperature_C'].values,
    'UV': weather_data['UV'].values,
    'EnergyProd-Wh': energy_production.values
})


# Compute daily energy production sums
df['Date'] = pd.to_datetime(df['Date'])  # Ensure datetime format
daily_sums = df.groupby('Date')['EnergyProd-Wh'].sum().reset_index()
daily_sums.rename(columns={'EnergyProd-Wh': 'EnergyProd-Wh-Daily-Sum'}, inplace=True)

# Merge back into original DataFrame
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

    # Apply classification
df['Cloud_Index'] = df.apply(classify_cloud, axis=1)

    # Optional: Preview
print(df['Cloud_Index'].value_counts())
print(df.head(100))

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

# 💾 Export to CSV
df.to_csv('ist_.csv', index=False)

# 🖥️ Preview
print(df.head())