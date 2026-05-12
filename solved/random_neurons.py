import random
from itertools import combinations
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ActivationFunction:
    def __init__(self, act_function):
        self.act_function = act_function

    def activate(self, v):
        return self.act_function(v)

class Perceptron:
    def __init__(self, act_function, n_inputs, n_outputs):
        self.act_function = act_function
        self.n_inputs = n_inputs + 1  # +1 por causa do bias
        self.n_outputs = n_outputs
        self.synaptic_weights = np.random.rand(n_outputs, self.n_inputs)

    def output(self, inputs):
        v = np.dot(self.synaptic_weights, inputs)
        return np.array(list(map(self.act_function.activate, v)))

    def learn(self, y_t, y_e, x, alpha):
        error = y_t - y_e
        self.synaptic_weights += alpha * error.reshape(-1, 1) * x

    def predict(self, X):
        X = self._add_bias(X)
        return np.array([self.output(x) for x in X])

    def _add_bias(self, X):
        bias = np.ones((X.shape[0], 1))
        return np.hstack((X, bias))

    def fit(self, X, y, alpha=0.01, epochs=50):
        X = np.array(X)
        y = np.array(y)

        X = self._add_bias(X)

        indices = list(range(len(X)))
        e_epoch = []

        for epoch in range(epochs):
            shuffle(indices)
            errors = []

            for k in indices:
                y_e = self.output(X[k])
                error = y[k] - y_e
                errors.append(error)

                self.learn(y[k], y_e, X[k], alpha)

            e_epoch.append(np.mean(np.abs(errors)))

        return e_epoch
    
class Neuron:
    """
    Basic neuron model used to build a single
    processing unit.
    """
    def __init__(self, act_func, d_act_func, pre_act=0, post_act=0):
        self.activation_function = act_func
        self.activation_function_derivative = d_act_func
        self.pre_activation = pre_act
        self.post_activation = post_act

    def process(self,v):
        self.pre_activation = v
        self.post_activation = self.activation_function(v)
        return self.post_activation

    def process_d(self):
        return self.activation_function_derivative(self.pre_activation)

class Layer:
    """
    Model for a single processing Layer.
    Once a layer is created, an additional input is always created
    to include bias.
    """
    def __init__( self, dimension, neuron_model, act_func, d_act_func ):
        """
        dimension: number of neurons in the layer
        No initial condition is passed to neurons. To change it, each
        neuron must be accessed through self.neurons array.
        """
        self.dimension = dimension
        self.neuron_model = neuron_model
        self.neurons = np.empty(dimension, dtype=object)
        self.pre_activation = np.zeros((dimension,1))
        self.post_activation = np.zeros((dimension,1))
        self.local_derivatives = np.zeros((dimension,1))
        for neuron_index in range(dimension):
            self.neurons[neuron_index] = neuron_model(act_func, d_act_func)

    def process(self, v, learn=False):
        """
        v is the vector (dimension x1), which corresponds
        to the local field for each neuron
        """
        self.pre_activation = np.array(v).reshape(-1,1) ## reshape(-1,1) -> any number of lines and only one column
        output = []
        deriv = []
        for i,neuron in enumerate(self.neurons):
            output.append(neuron.process(v[i]))
            if learn:
                deriv.append(neuron.process_d())
        self.post_activation = np.array(output).reshape(-1,1)
        if learn:
            self.local_derivatives = np.array(deriv).reshape(-1,1)
        return np.array(output).reshape(-1,1)

    def process_d(self):
        output = []
        for neuron in self.neurons:
            output.append(neuron.process_d()[0])
        self.local_derivatives = np.array(output).reshape(-1,1)
        return self.local_derivatives

class FFNeuralNetwork:
    """
      A class that models a feedforward neural network
    """
    def __init__(self, topology, layers, W0 = None, zero_init = False, rand_seed = 0):
        """
         topology: array that contains the number of neurons
         per layer, including input layer (i.e., a network
         with topology [3,2,1] contains three inputs, 2
         neurons in the hidden layer and one output layer.
        """
        self.topology = topology        # defines a dense feedforward NN
        self.n_layers = len(topology)-1 # number of processing layers
        self.layers = layers
        self.weights = W0
        self.e_epochs = []
        self.mse_epochs = []
        self.fold_errors = []
        if self.weights is None: # no initialization was provided
            self.weights = []
            if not zero_init: #random initialization is performed
                np.random.seed(rand_seed)
                for i in range(len(topology)-1):
                    #negative and positive initial weights:
                    self.weights.append(np.random.uniform(low=-1.0, high=1.0, size=(self.topology[i+1], self.topology[i] + 1)))
                    # only positive initial weights:
                    # self.weights.append(np.random.rand(self.topology[i+1],self.topology[i]+1)) ## bias is taken into account here
            else:
                for i in range(len(topology)-1):
                    self.weights.append(np.zeros((self.topology[i+1],self.topology[i]+1))) ## bias is taken into account here

    def process(self, x, learn=False):
        """
        Performs the forward propagation and gets neural network output from
        vector input x
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                x_in = np.vstack( (x,[[1]] )) # here we stack the bias input for the first layer
            else:
                x_in = np.vstack( [self.layers[i-1].post_activation, [[1]]])
            v = self.weights[i] @ x_in
            layer.process(v, learn)
        return self.layers[self.n_layers-1].post_activation

    def backprop(self, x, y_d, eta):
        """
        Here we implement the backpropagation algorithm.
        It is assumed that x and y_d are both column vectors and
        it is also assumed that the layers were appropriately
        initialized.

        This algorithm is implemented for a single example learning
        and it can be easily wrapped in a version that is applicable
        to some training set.

        eta: learning rate
        """

        # First we propagate through the network
        output = self.process(x, learn=True)
        # Then we compute the output error vector
        error_vector = y_d - output

        # Now we compute the gradients for the output and hidden layers, in
        # reverse order
        grad_indices = list(range(self.n_layers))
        grad_indices.reverse()
        local_grads = []
        for i in range(self.n_layers):
            local_grads.append(np.zeros((self.topology[i+1],1)))
        for l_idx in grad_indices:
            if l_idx == (self.n_layers - 1) : #output layer
                 #local gradient for the output layer
                 delta_k = error_vector * self.layers[self.n_layers-1].local_derivatives  #element-wise operation
                 local_grads[l_idx] = delta_k
            else:
                # delta_k_1 = phi'(v_k_1) *  (W_k^T. delta_k )
                delta_k_1 =  self.layers[l_idx].local_derivatives * (self.weights[(l_idx + 1)][:, :-1].T @ local_grads[(l_idx+1)] )
                local_grads[l_idx] = delta_k_1
        # now we can compute the appropriate weight corrections:
        for i in range(self.n_layers):
            if i == 0: #input layer
                self.weights[i] = self.weights[i] + eta * local_grads[i] @ np.vstack((x,[[1]])).T
            else:
                self.weights[i] = self.weights[i] + eta * local_grads[i] @ np.vstack( (self.layers[i-1].post_activation, [[1]])).T
        #Return the error just to measure the training process effectiveness:
        return error_vector

    def fit(self, x, y, learning_rate = 0.05, n_epochs = 500):
        self.e_epochs = []
        self.mse_epochs = []
        indices = list(range(len(x)))

        for epoch in range(n_epochs):
            e_point = []
            e_sum = 0
            np.random.shuffle(indices)

            for k in indices:
                x_i = x[k].reshape(-1,1)   # ✅ agora correto
                y_i = y[k].reshape(-1,1)

                error = self.backprop(x_i, y_i, learning_rate)
                err_val = error[0][0]
                e_sum += error[0][0]**2
                e_point.append(err_val)

            self.e_epochs.append(np.mean(np.abs(e_point)))
            self.mse_epochs.append(e_sum / len(x))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Erro: {self.e_epochs[-1]}")

        return self

    def plot_error(self):
        plt.plot(self.e_epochs, color="black")
        plt.title("Erro médio de classificação")
        plt.xlabel("Época de treinamento")
        plt.ylabel("Erro")
        plt.grid()
        plt.show()

    def plot_mse(self):
        plt.figure()
        plt.plot(self.mse_epochs, color="black")
        plt.title("Erro Quadrático Médio (MSE)")
        plt.xlabel("Época de treinamento")
        plt.ylabel("MSE")
        plt.grid()
        plt.show()

class RandomNeurons:

    def __init__(self, model_factory, type_ = "perceptron", L = 3, M = 100, a = 0.01, seed = 42):
        self.L = L
        self.epochs = M
        self.alpha = a
        self.seed = seed
        self.model_factory = model_factory
        self.ensemble = None
        self.type_ = type_
    
    def bootstrap(self, X, y):
        """Feature Sampling."""
        n_samples, n_features = X.shape

        results = []

        for _ in range(self.L):
            # Bootstrap das linhas
            row_idx = np.random.choice(n_samples, size=n_samples, replace=True)

            # Sorteia k (número de features)
            k = np.random.randint(1, n_features + 1)

            # Seleciona k features sem reposição
            col_idx = np.random.choice(n_features, size=k, replace=False)

            # Subconjuntos
            X_b = X[row_idx][:, col_idx]
            y_b = y[row_idx]

            results.append({
                "row_idx": row_idx,
                "col_idx": col_idx,
                "x": X_b,
                "y": y_b
            })

        return results
  
    def fit(self, X, y):
        """Treina um ensemble de modelos usando bootstrap e feature sampling usando a estratégia OvR."""
        d_l = self.bootstrap(X, y)
        n_classes = len(np.unique(y))

        ensemble = []

        for dl in d_l:
            X = dl["x"]
            y = dl["y"]

            models_per_replica = {}

            for c in range(n_classes):
                # binariza
                y_binary = (y == c).astype(int).reshape(-1, 1)

                model = self.model_factory(X.shape[1], 1)
                model.fit(X, y_binary, self.alpha, self.epochs)

                models_per_replica[c] = model

            ensemble.append({
                "models": models_per_replica,
                "col_idx": dl["col_idx"]
            })

        self.ensemble = ensemble
        return self.ensemble

    def predict(self, X):
        X = np.array(X)
        preds = []

        for x in X:
            votes = []

            for rep in self.ensemble:
                models = rep["models"]
                col_idx = rep["col_idx"]

                scores = []

                for c in sorted(models.keys()):
                    model = models[c]

                    x_sub = x[col_idx]

                    if self.type_ == "perceptron":
                        x_input = np.append(x_sub, 1)
                        y_hat = model.output(x_input)
                        scores.append(y_hat[0])

                    elif self.type_ == "mlp":
                        x_input = x_sub.reshape(-1,1)
                        y_hat = model.process(x_input)
                        scores.append(y_hat[0][0])

                pred = np.argmax(scores)
                votes.append(pred)

            final = max(set(votes), key=votes.count)
            preds.append(final)

        return np.array(preds)

    def _plot_decision_lines_perceptron(self, X, y):
        plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolors='k')

        x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)

        for rep in self.ensemble:
            models = rep["models"]
            col_idx = rep["col_idx"]

            # só plota se tiver exatamente 2 features
            if len(col_idx) != 2:
                continue

            for c, model in models.items():
                w = model.synaptic_weights[0]

                if len(w) != 3:
                    continue  # precisa de 2 features + bias

                w1, w2, b = w

                if w2 == 0:
                    continue

                y_vals = -(w1 * x_vals + b) / w2
                plt.plot(x_vals, y_vals, '--', alpha=0.5)

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Linhas de decisão dos perceptrons")
        plt.show()

    def _plot_decision_lines_mlp(self, X, y):
        plt.figure(figsize=(8,6))

        # dados
        plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolors='k')

        x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
        y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]

        # percorre ensemble
        for rep in self.ensemble:
            models = rep["models"]
            col_idx = rep["col_idx"]

            for c, model in models.items():
                Z = []

                for point in grid:
                    x_sub = point[col_idx].reshape(-1,1)
                    out = model.process(x_sub)
                    Z.append(out[0][0])

                Z = np.array(Z).reshape(xx.shape)

                # curva de decisão (nível 0)
                plt.contour(xx, yy, Z, levels=[0], linewidths=1, alpha=0.5)

        plt.title("Curvas de decisão (MLP)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

    def plot_decision_lines(self, X, y):
        if self.type_ == "perceptron":
            self._plot_decision_lines_perceptron(self.ensemble, X, y)
        elif self.type_ == "mlp":
            self._plot_decision_lines_mlp(self.ensemble, X, y)

    def plot_boundary(self, X, y):
        # limites do gráfico
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

        # grade de pontos
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )

        # transforma grid em lista de pontos
        grid = np.c_[xx.ravel(), yy.ravel()]

        # predição do ensemble
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)

        # plot da região
        plt.figure(figsize=(8,6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

        # plot dos dados reais
        plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolors='k')

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Fronteira de decisão do ensemble")
        plt.show()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def plot_confusion(self, y_test, y_pred, classes):
        print("Acurácia:", accuracy_score(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5,4))
        plt.imshow(cm, cmap=plt.cm.Blues)

        plt.xticks(np.arange(len(classes)), classes)
        plt.yticks(np.arange(len(classes)), classes)

        # Labels dos eixos
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title("Matriz de Confusão")

        # Mostrar valores dentro das células
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")

        plt.colorbar()
        plt.tight_layout()
        plt.show()
