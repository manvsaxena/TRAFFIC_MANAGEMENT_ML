from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Function to convert datetime to POSIX time
def posix_time(dt):
    return (dt - datetime(1970, 1, 1)) / timedelta(seconds=1)

# Read and preprocess the data
data = pd.read_csv('static/Train.csv')
data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)

# Create features for the last n hours of traffic volume
last_n_hours = [1, 2, 3, 4, 5, 6]
for n in last_n_hours:
    data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)

# Drop rows with missing values
data = data.dropna().reset_index(drop=True)

# Encode holiday as binary
data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
data['is_holiday'] = data['is_holiday'].astype(int)

# Convert date_time to datetime object and extract time-based features
data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].dt.hour
data['month_day'] = data['date_time'].dt.day
data['weekday'] = data['date_time'].dt.weekday + 1
data['month'] = data['date_time'].dt.month
data['year'] = data['date_time'].dt.year

# Save processed data for further use
data.to_csv("traffic_volume_data.csv", index=False)

# Sampling the data for model training
sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# Load the cleaned data
data = pd.read_csv("traffic_volume_data.csv")

# Ensure we don't sample more rows than available
data = data.sample(min(10000, len(data)), replace=False).reset_index(drop=True)

# Define feature columns and target
numeric_columns = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month']
features = numeric_columns
target = ['traffic_volume']

# Prepare features and target variable
X = data[features]
y = data[target]

# Normalize features and target
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y).flatten()

# Initialize and train the model
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)

# Initialize Flask app
app = Flask(__name__, static_url_path='')

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate and get form data
        is_holiday = request.form.get('isholiday')
        temperature = request.form.get('temperature')
        time = request.form.get('time')
        date = request.form.get('date')
        x0 = request.form.get('x0')
        x1 = request.form.get('x1')

        # Handle missing or invalid inputs with default values
        is_holiday = 1 if is_holiday == 'yes' else 0
        temperature = int(temperature) if temperature.isdigit() else 0
        weekday = 0  # Default value for weekday if not provided
        hour = int(time[:2]) if time and len(time) > 1 else 0
        month_day = int(date[8:]) if date and len(date) > 7 else 1
        year = int(date[:4]) if date and len(date) > 3 else 2024
        month = int(date[5:7]) if date and len(date) > 4 else 1

        # Check if weather data is provided
        if x0 not in ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Thunderstorm']:
            x0 = 'Clear'  # Default weather type
        if x1 not in ['Sky is Clear', 'broken clouds', 'drizzle', 'few clouds', 'fog', 'haze', 'heavy intensity drizzle',
                      'heavy intensity rain', 'heavy snow', 'light intensity drizzle', 'light intensity shower rain',
                      'light rain', 'light rain and snow', 'light shower snow', 'light snow', 'mist', 'moderate rain',
                      'overcast clouds', 'proximity shower rain', 'proximity thunderstorm', 'proximity thunderstorm with drizzle',
                      'proximity thunderstorm with rain', 'scattered clouds', 'shower drizzle', 'sky is clear', 'sleet',
                      'smoke', 'snow', 'thunderstorm', 'thunderstorm with heavy rain', 'thunderstorm with light drizzle',
                      'thunderstorm with light rain', 'thunderstorm with rain', 'very heavy rain']:
            x1 = 'Sky is Clear'  # Default weather description

        # One-hot encode weather features
        x0_values = ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Thunderstorm']
        x1_values = ['Sky is Clear', 'broken clouds', 'drizzle', 'few clouds', 'fog', 'haze', 'heavy intensity drizzle',
                     'heavy intensity rain', 'heavy snow', 'light intensity drizzle', 'light intensity shower rain',
                     'light rain', 'light rain and snow', 'light shower snow', 'light snow', 'mist', 'moderate rain',
                     'overcast clouds', 'proximity shower rain', 'proximity thunderstorm', 'proximity thunderstorm with drizzle',
                     'proximity thunderstorm with rain', 'scattered clouds', 'shower drizzle', 'sky is clear', 'sleet',
                     'smoke', 'snow', 'thunderstorm', 'thunderstorm with heavy rain', 'thunderstorm with light drizzle',
                     'thunderstorm with light rain', 'thunderstorm with rain', 'very heavy rain']

        # One-hot encode the weather features for prediction
        x0_encoded = {f'x0_{i}': 1 if i == x0 else 0 for i in x0_values}
        x1_encoded = {f'x1_{i}': 1 if i == x1 else 0 for i in x1_values}

        # Create final feature vector: only include numeric features for training
        final_features = [
            is_holiday, temperature, weekday, hour, month_day, year, month
        ] + list(x0_encoded.values()) + list(x1_encoded.values())

        # Ensure the feature vector has the same number of features as the training data
        final_features = final_features[:len(features)]  # Ensure it has only 7 numeric features

        # Predict traffic volume
        prediction = regr.predict([final_features])
        predicted_traffic = y_scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]

        return render_template('output.html', data1=request.form, data2=predicted_traffic)
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
