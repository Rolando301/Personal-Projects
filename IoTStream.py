import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Step 1: Simulate IoT Sensor Data
# -----------------------------
num_sensors = 5
num_minutes = 60  # simulate 1 hour
start_time = datetime.now()

data = []
for minute in range(num_minutes):
    timestamp = start_time + timedelta(minutes=minute)
    for sensor_id in range(1, num_sensors + 1):
        temperature = np.random.normal(loc=22, scale=2)
        humidity = np.random.normal(loc=40, scale=5)
        data.append([timestamp, f'Sensor{sensor_id}', round(temperature, 2), round(humidity, 2)])

df = pd.DataFrame(data, columns=['timestamp', 'sensor_id', 'temperature', 'humidity'])

# -----------------------------
# Step 2: Transform Data
# -----------------------------
df = df.sort_values(by=['sensor_id', 'timestamp'])
df['temp_rolling_mean'] = df.groupby('sensor_id')['temperature'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['humidity_diff'] = df.groupby('sensor_id')['humidity'].transform(lambda x: x.diff().fillna(0))

# -----------------------------
# Step 3: Anomaly Detection
# -----------------------------
def detect_anomalies(series, window=5, threshold=2):
    rolling_mean = series.rolling(window, min_periods=1).mean()
    rolling_std = series.rolling(window, min_periods=1).std().fillna(0)
    anomalies = (series - rolling_mean).abs() > threshold * rolling_std
    return anomalies.astype(int)

df['temp_anomaly'] = df.groupby('sensor_id')['temperature'].transform(detect_anomalies)
df['humidity_anomaly'] = df.groupby('sensor_id')['humidity'].transform(detect_anomalies)

# Combined anomalies
df['combined_anomaly'] = ((df['temp_anomaly'] == 1) & (df['humidity_anomaly'] == 1)).astype(int)

# Consecutive anomalies
def consecutive_anomalies(series, threshold=3):
    count = 0
    result = []
    for val in series:
        if val == 1:
            count += 1
        else:
            count = 0
        result.append(1 if count >= threshold else 0)
    return pd.Series(result, index=series.index)

df['consecutive_anomaly'] = df.groupby('sensor_id')['combined_anomaly'].transform(consecutive_anomalies)

# -----------------------------
# Step 4: SQLite for SQL Queries
# -----------------------------
conn = sqlite3.connect(':memory:')
df.to_sql('sensor_readings', conn, index=False, if_exists='replace')

# Combined anomalies SQL query
query_combined = """
SELECT timestamp, sensor_id, temperature, humidity, combined_anomaly, consecutive_anomaly
FROM sensor_readings
WHERE combined_anomaly = 1
ORDER BY timestamp
LIMIT 10
"""
combined_anomalies = pd.read_sql_query(query_combined, conn)
print("Combined anomalies (temperature + humidity):\n", combined_anomalies)

# Hourly averages SQL query
df['hour'] = df['timestamp'].dt.floor('H')
df.to_sql('sensor_readings', conn, index=False, if_exists='replace')  # update with hour column

query_hourly = """
SELECT sensor_id, hour, AVG(temperature) as avg_temp, AVG(humidity) as avg_humidity
FROM sensor_readings
GROUP BY sensor_id, hour
ORDER BY sensor_id, hour
"""
hourly_avg = pd.read_sql_query(query_hourly, conn)
print("\nHourly average temperature and humidity per sensor:\n", hourly_avg)

# -----------------------------
# Step 5: Visualizations
# -----------------------------
sns.set(style="whitegrid")

# Temperature anomalies per sensor
plt.figure(figsize=(12,6))
for sensor in df['sensor_id'].unique():
    sensor_data = df[df['sensor_id'] == sensor]
    plt.plot(sensor_data['timestamp'], sensor_data['temperature'], label=f"{sensor} Temp")
    anomalies = sensor_data[sensor_data['temp_anomaly'] == 1]
    plt.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', s=50, label=f"{sensor} Anomaly")

plt.title('Temperature and Anomalies per Sensor')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Combined anomalies over time
plt.figure(figsize=(12,6))
combined_count = df.groupby('timestamp')['combined_anomaly'].sum()
combined_count.plot(kind='bar', color='orange')
plt.title('Number of Combined Anomalies Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Count of Combined Anomalies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Hourly average temperature per sensor
plt.figure(figsize=(12,6))
hourly_avg_plot = df.groupby(['hour','sensor_id'])[['temperature','humidity']].mean().reset_index()
sns.lineplot(data=hourly_avg_plot, x='hour', y='temperature', hue='sensor_id', marker='o')
plt.title('Hourly Average Temperature per Sensor')
plt.xlabel('Hour')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Step 6: Real-time Style Alerts
# -----------------------------
def alert_consecutive_anomalies(df, threshold=3):
    alerts = []
    for sensor in df['sensor_id'].unique():
        sensor_data = df[df['sensor_id'] == sensor]
        if sensor_data['consecutive_anomaly'].sum() > 0:
            alert_times = sensor_data[sensor_data['consecutive_anomaly'] == 1]['timestamp'].tolist()
            alerts.append((sensor, alert_times))
    return alerts

alerts = alert_consecutive_anomalies(df)

if alerts:
    print("\n⚠️  ALERT: Consecutive anomalies detected!")
    for sensor, times in alerts:
        print(f"- {sensor} has consecutive anomalies at times:")
        for t in times:
            print(f"    {t}")
else:
    print("\n✅ No consecutive anomalies detected.")
