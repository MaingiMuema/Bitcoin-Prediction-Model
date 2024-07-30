import numpy as np
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def predict_and_inverse_transform(model, X_train, X_test, scaler):
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    return train_predict, test_predict

def calculate_rmse(Y_train, Y_test, train_predict, test_predict, scaler):
    Y_train = scaler.inverse_transform([Y_train])
    Y_test = scaler.inverse_transform([Y_test])

    train_rmse = math.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
    test_rmse = math.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))

    return train_rmse, test_rmse

def plot_predictions(scaled_data, train_predict, test_predict, time_step):
    # Plot baseline and predictions
    plt.figure(figsize=(14, 5))
    plt.plot(scaled_data)
    plt.plot(np.arange(time_step, len(train_predict) + time_step), train_predict)
    plt.plot(np.arange(len(train_predict) + (time_step * 2) + 1, len(scaled_data) - 1), test_predict)
    plt.show()
