from tensorflow import keras
from tensorflow.keras import layers

def create_1layer_model(input_dim, hidden_dim):
    inputs = layers.Input(shape=(input_dim,))

    # Hidden layer with deterministic weights using a Dense layer and nonlinear activation function
    features = layers.Dense(hidden_dim, activation="sigmoid")(inputs)

    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
