from sklearn.linear_model import LinearRegression

def train_linear_regression(full_transform, car_train_set):
    car_train_set_attributes = car_train_set.drop('selling_price', axis=1)
    car_train_set_labels = car_train_set['selling_price']

    # Training
    car_train_set_attributes_prepared = full_transform.fit_transform(car_train_set_attributes)
    linear_reg = LinearRegression()
    linear_reg.fit(car_train_set_attributes_prepared, car_train_set_labels)
    return linear_reg