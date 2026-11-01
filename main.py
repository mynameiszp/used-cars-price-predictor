import pandas as pd
car_data = pd.read_csv('dataset.csv')
car_data = car_data.drop(labels=['fuel', 'owner'], axis=1)
