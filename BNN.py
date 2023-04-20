import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n), 
                                                                   scale_diag=tf.ones(n))
            )
        ]
    )
    return prior_model
    
    # Define variational posterior weight distribution as multivariate Gaussian.
# The learnable parameters for this distribution are the means, variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def create_full_bnn_model(input_dim, hidden_units, kl_weight):
    inputs = layers.Input(shape=(input_dim,))
    features = layers.BatchNormalization()(inputs)

    # Create hidden layer with weight uncertainty using the DenseVariational layer
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=kl_weight,
            activation="sigmoid",
        )(features)

    outputs = tfp.layers.DenseVariational(
            units=1,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=kl_weight,
    )(features)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
