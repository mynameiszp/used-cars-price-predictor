import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def calculate_error(car_data, car_validation_set, full_transform, linear_reg):
    car_validation_set_attributes = car_validation_set.drop('selling_price', axis=1)
    car_validation_set_labels = car_validation_set['selling_price']

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