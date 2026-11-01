import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

car_data = pd.read_csv('dataset.csv')
car_data['brand'] = car_data["name"].str.split().str[0]
car_data = car_data.drop(labels=['name'], axis=1)
car_train_set, car_test_set = train_test_split(car_data, test_size=0.2, random_state=42)

attributes_category = ['owner', 'transmission', 'seller_type', 'brand', 'fuel']
attributes_numerical = ['year', 'km_driven']

full_transform = ColumnTransformer([
    ("cat", OneHotEncoder(), attributes_category),
    ("num", StandardScaler(), attributes_numerical)
])
car_train_set_attributes = car_train_set.drop('selling_price', axis=1)
car_train_set_labels = car_train_set['selling_price']
car_train_set_attributes_prepared = full_transform.fit_transform(car_train_set_attributes)
