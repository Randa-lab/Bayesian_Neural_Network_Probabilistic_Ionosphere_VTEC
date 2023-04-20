from tensorflow import keras

def run_experiment(model, loss, num_epochs, num_batch_size, X_train, y_train):
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    
    history=model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch)
    _, rmset = model.evaluate(X_train, y_train, verbose=0)
    print(f"Train RMSE: {round(rmset, 3)}")
