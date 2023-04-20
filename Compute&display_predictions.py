import numpy as np

def compute_predictions(model, X_data, iterations=100):
    predicted = []
    for _ in range(iterations):
        predicted.append(model(examples).numpy())
    predicted = np.concatenate(predicted, axis=1)
    return predicted

def display_predictions(predictions, targets):
    prediction_mean = np.mean(predictions, axis=1).tolist()
    prediction_min = np.min(predictions, axis=1).tolist()
    prediction_max = np.max(predictions, axis=1).tolist()
    prediction_range = (np.max(predictions, axis=1) - np.min(predictions, axis=1)).tolist()

    for idx in range(samples):
        print(
            f"Predictions mean: {round(prediction_mean[idx], 2)}, "
            f"min: {round(prediction_min[idx], 2)}, "
            f"max: {round(prediction_max[idx], 2)}, "
            f"range: {round(prediction_range[idx], 2)} - "
            f"Actual: {targets[idx]}"
        )
