from src.data_preparation import load_and_prepare_data, prepare_data_for_lstm
from src.model import build_lstm_model
from src.train import train_model
from src.evaluate import predict_and_inverse_transform, calculate_rmse, plot_predictions

# Load and prepare data
file_path = 'data/BTC-Hourly.csv'
column_name = 'close'
train_data, test_data, scaler = load_and_prepare_data(file_path, column_name)

# Prepare data for LSTM
time_step = 10
X_train, Y_train, X_test, Y_test = prepare_data_for_lstm(train_data, test_data, time_step)

# Build and train the model
model = build_lstm_model(time_step)
model = train_model(model, X_train, Y_train)

# Evaluate the model
train_predict, test_predict = predict_and_inverse_transform(model, X_train, X_test, scaler)
train_rmse, test_rmse = calculate_rmse(Y_train, Y_test, train_predict, test_predict, scaler)

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Plot the predictions
plot_predictions(scaler.inverse_transform(train_data), train_predict, test_predict, time_step)
