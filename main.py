import pandas as pd
from sklearn.model_selection import train_test_split
car_data = pd.read_csv('dataset.csv')
car_data = car_data.drop(labels=['fuel', 'owner'], axis=1)
car_train_set, car_test_set = train_test_split(car_data, test_size=0.2, random_state=42)

