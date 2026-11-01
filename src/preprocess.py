import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def load_data():
    car_data = pd.read_csv('data/dataset.csv')
    car_data['brand'] = car_data["name"].str.split().str[0]
    return car_data.drop(labels=['name'], axis=1)

def split_data(car_data):
    car_train_set, car_test_set = train_test_split(car_data, test_size=0.2, random_state=42)
    car_train_set, car_validation_set = train_test_split(car_train_set, test_size=0.25, random_state=42)
    return car_train_set, car_validation_set, car_test_set

def create_preprocessor():
    attributes_category = ['owner', 'transmission', 'seller_type', 'brand', 'fuel']
    attributes_numerical = ['year', 'km_driven']

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), attributes_category),
        ("num", StandardScaler(), attributes_numerical)
    ])