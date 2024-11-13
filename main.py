from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
from skopt.space import Real, Categorical, Integer
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load and preprocess data
data = pd.read_csv('static/Train.csv')
data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)

# Create features for traffic volume in the last n hours
last_n_hours = [1, 2, 3, 4, 5, 6]
for n in last_n_hours:
    data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
data = data.dropna().reset_index(drop=True)

# Encode holiday feature
data['is_holiday'] = data['is_holiday'].apply(lambda x: 1 if x != 'None' else 0)

# Extract date and time features
data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].dt.hour
data['month_day'] = data['date_time'].dt.day
data['weekday'] = data['date_time'].dt.weekday + 1
data['month'] = data['date_time'].dt.month
data['year'] = data['date_time'].dt.year
data.to_csv("traffic_volume_data.csv", index=False)

# Data visualization setup
sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# Reload processed data
data = pd.read_csv("traffic_volume_data.csv")

# Sample data to avoid taking more than available
sample_size = min(10000, len(data))
data = data.sample(sample_size).reset_index(drop=True)

# Define feature columns
label_columns = ['weather_type', 'weather_description']
numeric_columns = [
    'is_holiday', 'air_pollution_index', 'humidity', 'wind_speed', 
    'wind_direction', 'visibility_in_miles', 'dew_point', 'temperature', 
    'rain_p_h', 'snow_p_h', 'clouds_all', 'weekday', 'hour', 'month_day', 
    'year', 'month', 'last_1_hour_traffic', 'last_2_hour_traffic', 
    'last_3_hour_traffic'
]

# One-hot encode categorical features
ohe_encoder = OneHotEncoder()
x_ohehot = ohe_encoder.fit_transform(data[label_columns])
ohe_features = ohe_encoder.get_feature_names_out()
x_ohehot = pd.DataFrame(x_ohehot.toarray(), columns=ohe_features)

# Combine numeric and one-hot encoded features
data = pd.concat([data[['date_time']], data[['traffic_volume'] + numeric_columns], x_ohehot], axis=1)

# Plot traffic volume trends
metrics = ['month', 'month_day', 'weekday', 'hour']
fig = plt.figure(figsize=(8, 4 * len(metrics)))
for i, metric in enumerate(metrics):
    ax = fig.add_subplot(len(metrics), 1, i + 1)
    ax.plot(data.groupby(metric)['traffic_volume'].mean(), '-o')
    ax.set_xlabel(metric)
    ax.set_ylabel("Mean Traffic")
    ax.set_title(f"Traffic Trend by {metric}")
plt.tight_layout()
plt.show()

# Model setup and training
features = numeric_columns + list(ohe_features)
target = ['traffic_volume']
X = data[features]
y = data[target]

# Scale features
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y).flatten()

# Train MLPRegressor
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
print("MLPRegressor Predictions:", regr.predict(X[:10]))
print("Actual Values:", y[:10])

# Model parameter dictionary for Bayesian Optimization
Model_params = {
    MLPRegressor: {
        'hidden_layer_sizes': Integer(50, 400),
        'solver': Categorical(['sgd', 'adam']),
        'learning_rate_init': Real(0.001, 0.1)
    },
    SVR: {
        'C': Real(1e-1, 1e+1, prior='log-uniform'),
        'degree': Integer(1, 4),
        'kernel': Categorical(['linear', 'rbf'])
    }
}

# Bayesian Optimization function
def bayes_opt(Model, params: dict, X, y, n_iter=5, scoring='r2', cv=3):
    print(f"- Bayes Optimizing: {Model.__name__}")
    opt = BayesSearchCV(
        Model(),
        params,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=1
    )
    opt.fit(X, y)
    print(f"Best Parameters for {Model.__name__}: {opt.best_params_}")
    return opt, opt.best_params_, opt.best_score_

# Optimize parameters for each model
best_params_dic = {}
for Model, params in Model_params.items():
    opt, best_params_, best_score_ = bayes_opt(Model, params, X, y)
    best_params_dic[Model] = dict(best_params_)

print("Best Parameters Dictionary:", best_params_dic)
