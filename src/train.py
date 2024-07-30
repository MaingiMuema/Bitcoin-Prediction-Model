def train_model(model, X_train, Y_train, batch_size=1, epochs=1):
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
    return model
