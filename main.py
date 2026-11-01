import src.preprocess as preprocess
import src.train as train
import src.evaluate as evaluate

data = preprocess.load_data()
car_train_set, car_validation_set, car_test_set = preprocess.split_data(data)
preprocessor = preprocess.create_preprocessor()

linear_reg_model = train.train_linear_regression(preprocessor, car_train_set)

evaluate.calculate_error(car_validation_set, car_test_set, preprocessor, linear_reg_model)
