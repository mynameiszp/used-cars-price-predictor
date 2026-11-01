import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

car_data = pd.read_csv('dataset.csv')
car_data['brand'] = car_data["name"].str.split().str[0]
car_data = car_data.drop(labels=['name'], axis=1)
car_train_set, car_test_set = train_test_split(car_data, test_size=0.2, random_state=42)
car_train_set, car_validation_set = train_test_split(car_train_set, test_size=0.25, random_state=42)

attributes_category = ['owner', 'transmission', 'seller_type', 'brand', 'fuel']
attributes_numerical = ['year', 'km_driven']

full_transform = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), attributes_category),
    ("num", StandardScaler(), attributes_numerical)
])
car_train_set_attributes = car_train_set.drop('selling_price', axis=1)
car_validation_set_attributes = car_validation_set.drop('selling_price', axis=1)
car_train_set_labels = car_train_set['selling_price']
car_validation_set_labels = car_validation_set['selling_price']

# Training
car_train_set_attributes_prepared = full_transform.fit_transform(car_train_set_attributes)
linear_reg = LinearRegression()
linear_reg.fit(car_train_set_attributes_prepared, car_train_set_labels)

# Assessing with validation data
car_validation_set_transform = full_transform.transform(car_validation_set_attributes)
car_validation_set_predictions_labels = linear_reg.predict(car_validation_set_transform)

# Error calculation
rmse_error = np.sqrt(mean_squared_error(car_validation_set_labels, car_validation_set_predictions_labels))
mae_error = mean_absolute_error(car_validation_set_labels, car_validation_set_predictions_labels)
print('mean: ', car_data["selling_price"].mean())
print('median: ', car_data["selling_price"].median())
print('MAE: ', mae_error)
print('RMSE: ', rmse_error)
