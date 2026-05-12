import time

import numpy as np
import optuna
import pandas as pd
import pystac_client
from extraction_data_methods import (SGolay, encode_label, extract_bands,
                                     extract_features, interpolate_,
                                     normalize_, smooth_, std_)
from mlp_multiclass import (FFNeuralNetwork, He_, Layer, Neuron, Xavier,
                            d_relu, d_softmax, d_tanh, relu, softmax, tanh)
from optuna.visualization.matplotlib import (plot_optimization_history,
                                             plot_param_importances)
from sklearn.model_selection import train_test_split


def objective(trial):
    # ==================================================
    # HYPERPARAMETERS
    # ==================================================
    hidden1 = trial.suggest_int("hidden1", 16, 128)
    hidden2 = trial.suggest_int("hidden2", 8, 64)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    l2_lambda = trial.suggest_float("l2_lambda", 1e-6, 1e-2, log=True)
    activation_name = trial.suggest_categorical("activation", ["relu", "tanh"])
    init_name = trial.suggest_categorical("initializer", ["he", "xavier"])

    # ==================================================
    # ACTIVATION
    # ==================================================
    if activation_name == "relu":
        act = relu
        d_act = d_relu
    else:
        act = tanh
        d_act = d_tanh

    # ==================================================
    # INITIALIZATION
    # ==================================================
    if init_name == "he":
        initializer = He_
    else:
        initializer = Xavier

    # ==================================================
    # NETWORK
    # ==================================================
    input_size = X_train.shape[1]
    topology = [input_size, hidden1, hidden2, 9]
    layers = [
        Layer(hidden1, Neuron, act, d_act),
        Layer(hidden2, Neuron, act, d_act),
        Layer(9, Neuron, softmax, d_softmax, multiclass=True)
    ]
    nn = FFNeuralNetwork(topology, layers, multiclass=True, method_init=initializer, lambda_l2=l2_lambda)

    # ==================================================
    # TRAIN
    # ==================================================
    nn.fit_with_validation(X_train, y_train, X_val, y_val, learning_rate=learning_rate, n_epochs=100)

    # ==================================================
    # VALIDATION
    # ==================================================
    result = nn.test(X_test, y_test)
    metrics = result["metrics"]["accuracy"]
    return metrics

# Start the timer
start = time.perf_counter()

samples = pd.read_csv('./samples/samples_mt_time_series.csv')

# get bands description
service = pystac_client.Client.open("https://data.inpe.br/bdc/stac/v1/")
collection = service.get_collection('mod13q1-6.1').to_dict()
bands_description = collection['properties']['eo:bands']

bands = ["NDVI", "EVI", "NIR_reflectance"]
sgolay = SGolay(7, 3)

samples = extract_bands(samples, bands)
samples_norm = normalize_(samples, bands_description)
samples_inter = interpolate_(samples_norm, bands_description)
samples_smoothed = smooth_(samples_inter, sgolay)

dataset = samples_smoothed.copy()

X = []
y = []

for _, row in dataset.iterrows():
    X.append(extract_features(row["time_series"]))
    y.append(encode_label(samples, row["label"]))

X = np.array(X)
y = np.array(y)

# Example usage with your variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = (X_train - X_train.mean(axis=0)) / std_(X_train)
X_val = (X_val - X_val.mean(axis=0)) / std_(X_val)
X_test = (X_test - X_test.mean(axis=0)) / std_(X_test)
study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=50)

print(study.best_trial)
print(study.best_params)

# End the timer
end = time.perf_counter()

print(f"Elapsed time: {end - start:.6f} seconds")

plot_optimization_history(study)
plot_param_importances(study)

# Trial 16
# {'hidden1': 115, 'hidden2': 46, 'learning_rate': 0.008884237205192994, 'l2_lambda': 0.002587729914242751, 'activation': 'relu', 'initializer': 'he'}
# Elapsed time: 2828.104090 seconds
