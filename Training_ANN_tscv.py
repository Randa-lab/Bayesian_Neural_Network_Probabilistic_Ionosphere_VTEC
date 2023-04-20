from tensorflow import keras

#training ANN with time-series cross-validation (tscv)

from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=20)

learning_rate = 0.001

def run_experiment(model, loss, num_epochs, num_batch_size, train_X, train_y):
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    count = 1
    xtrain_err = 0
    xval_err = 0
    splits=20 

    for train_index, test_index in tscv.split(train_X, train_y):
      X_train, X_val =train_X[train_index], train_X[test_index]
      y_train, y_val = train_y[train_index], train_y[test_index]
      
      history=model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch, validation_data=(X_val, y_val))
      _, rmset = model.evaluate(X_train, y_train, verbose=0)
      print('Fold {}'.format(count))
      print(f"Train RMSE: {round(rmset, 3)}")
      
      _, rmsev = model.evaluate(X_val, y_val, verbose=0)
      print(f"Valid RMSE: {round(rmsev, 3)}")

      xtrain_err+=rmset
      xval_err+=rmsev
      
      count = count + 1
    
    train_rmse = xtrain_err/splits
    val_rmse = xval_err/splits
    print ('Average RMSE on train data', round(train_rmse,3))
    print ('Average RMSE on val data', round(val_rmse,3))
