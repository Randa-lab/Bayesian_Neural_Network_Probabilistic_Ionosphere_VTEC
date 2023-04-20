# Bayesian_Neural_Network_for_Probabilistic_Ionosphere_VTEC
Bayesian neural network models for probabilistic VTEC forecasting with 95% confidence,  from the paper "Uncertainty Quantification for Machine Learning-based Ionosphere and Space Weather Forecasting" by Natras et al., submitted to the Space Weather Jornal, AGU.
Two Bayesian neural network models were developed for VTEC forecasting with 95% confidence intervals. In both models, the deterministic network parameters (weights) are replaced by probability distributions of these weights. However, the first model only describes the uncertainty in the weights and estimates the model uncertainty, while the second model also estimate the data uncertainty by providing probabilistic output estimate via the negative log-likelihood (NLL) loss.
