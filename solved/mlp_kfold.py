import matplotlib.pyplot as plt
import numpy as np


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
        self.fold_val_mse_history = []
        self.fold_errors = []
        self.fold_precision = []
        self.fold_recall = []
        self.fold_f1 = []
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
    
    def compute_metrics(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (TP + TN) / (TP + TN + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        return accuracy, precision, recall, f1

    def fit(self, x, y, learning_rate = 0.05, n_epochs = 500):
        self.e_epochs = []
        self.mse_epochs = []
        indices = list(range(len(x)))

        for epoch in range(n_epochs):
            e_point = []
            e_sum = 0
            np.random.shuffle(indices)

            for k in indices:
                x_i = x[k].reshape(-1,1)   # agora correto
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
    
    def fit_with_val(self, X_train, y_train, X_val, y_val, learning_rate=0.05, n_epochs=500):
        self.mse_epochs = []
        self.val_mse_epochs = []
    
        indices = list(range(len(X_train)))
    
        for epoch in range(n_epochs):
            e_sum = 0
            np.random.shuffle(indices)
    
            # TREINO
            for k in indices:
                x_i = X_train[k].reshape(-1,1)
                y_i = y_train[k].reshape(-1,1)
    
                error = self.backprop(x_i, y_i, learning_rate)
                e_sum += error[0][0]**2
    
            train_mse = e_sum / len(X_train)
            self.mse_epochs.append(train_mse)
    
            val_errors = []
    
            for i in range(len(X_val)):
                x_i = X_val[i].reshape(-1,1)
                y_true = y_val[i]
    
                y_pred = self.process(x_i)[0][0]
                val_errors.append((y_true - y_pred)**2)
    
            val_mse = np.mean(val_errors)
            self.val_mse_epochs.append(val_mse)
    
            if epoch % 50 == 0:
                print(f"Epoch {epoch} | Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f}")
    
        return self

    def fit_by_folds(self, folds, learning_rate=0.05, n_epochs=500, threshold = 0.5):
        self.fold_errors = []
        self.fold_mse = []
        self.fold_accuracies = []
        self.fold_mse_history = []

        # salva pesos iniciais
        initial_weights = [w.copy() for w in self.weights]

        for fold_idx in folds:
            print(f"\n=== Fold {fold_idx} ===")

            X_train = folds[fold_idx]["X_train"]
            y_train = folds[fold_idx]["y_train"]
            X_val = folds[fold_idx]["X_val"]
            y_val = folds[fold_idx]["y_val"]

            # NORMALIZA (usando apenas treino)
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            self.scaler_mean = mean
            self.scaler_std = std
            std[std == 0] = 1

            X_train_norm = (X_train - mean) / std
            X_val_norm = (X_val - mean) / std

            # RESET dos pesos
            self.weights = [w.copy() for w in initial_weights]

            # TREINO
            self.fit_with_val(X_train_norm, y_train, X_val_norm, y_val, learning_rate, n_epochs)
            self.fold_mse_history.append(self.mse_epochs.copy())
            self.fold_val_mse_history.append(self.val_mse_epochs.copy())

            # VALIDAÇÃO
            errors = []
            squared_errors = []
            y_preds = []

            for i in range(len(X_val_norm)):
                x_i = X_val_norm[i].reshape(-1,1)
                y_true = y_val[i]

                y_pred = self.process(x_i)[0][0]

                # erro contínuo
                error = y_true - y_pred
                errors.append(abs(error))
                squared_errors.append(error**2)

                # classificação (threshold)
                pred_class = 1 if y_pred >= threshold else 0
                y_preds.append(pred_class)

            # métricas
            mean_error = np.mean(errors)
            mse = np.mean(squared_errors)

            self.fold_errors.append(mean_error)
            self.fold_mse.append(mse)
            
            acc, precision, recall, f1 = self.compute_metrics(y_val, y_preds)

            self.fold_accuracies.append(acc)
            self.fold_precision.append(precision)
            self.fold_recall.append(recall)
            self.fold_f1.append(f1)

            print(f"Erro médio (val): {mean_error:.4f}")
            print(f"MSE (val): {mse:.4f}")
            print(f"Acurácia (val): {acc:.4f}")
            print(f"Precisão (val): {precision:.4f}")
            print(f"Recall (val): {recall:.4f}")
            print(f"F1-score (val): {f1:.4f}")

        print("\n=== Resultado Final ===")
        print(f"Acurácia média: {np.mean(self.fold_accuracies):.4f}")
        print(f"Precisão média: {np.mean(self.fold_precision):.4f}")
        print(f"Recall médio: {np.mean(self.fold_recall):.4f}")
        print(f"F1-score médio: {np.mean(self.fold_f1):.4f}")

        return self

    def predict(self, X, threshold=0.5):
        y_probs = []
        y_preds = []
    
        for i in range(len(X)):
            x_i = X[i].reshape(-1, 1)
            y_pred = self.process(x_i)[0][0]
    
            y_probs.append(y_pred)
            y_preds.append(1 if y_pred >= threshold else 0)
    
        return np.array(y_probs), np.array(y_preds)

    def test(self, X_test, y_test, threshold=0.5, scaled = False):
        X_test_norm = X_test
        # normalização usando treino
        if not scaled:
            X_test_norm = (X_test - self.scaler_mean) / self.scaler_std
    
        # predição
        y_probs, y_preds = self.predict(X_test_norm, threshold)
    
        # métricas
        acc, precision, recall, f1 = self.compute_metrics(y_test, y_preds)
    
        print("\n=== Avaliação no conjunto externo ===")
        print(f"Acurácia: {acc:.4f}")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
    
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "y_pred": y_preds,
            "y_prob": y_probs
        }

    def confusion_matrix(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
    
        return np.array([[TN, FP], [FN, TP]])

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = self.confusion_matrix(y_true, y_pred)
    
        plt.figure(figsize=(6,5))
        plt.imshow(cm, cmap='Blues')
    
        plt.title("Matriz de Confusão")
        plt.colorbar()
    
        classes = [str(i) for i in range(cm.shape[0])]
        tick_marks = np.arange(len(classes))
    
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
    
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]}",
                         ha="center", va="center",
                         color="white" if cm[i, j] > cm.max()/2 else "black")
    
        plt.ylabel("Real")
        plt.xlabel("Predito")
    
        plt.tight_layout()
        plt.grid(False)
        plt.show()

    def plot_error(self):
        plt.plot(self.e_epochs, color="black")
        plt.title("Erro médio de classificação")
        plt.xlabel("Época de treinamento")
        plt.ylabel("Erro")
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.8)
        plt.show()

    def plot_train_val_mse(self):    
        plt.figure()
        plt.plot(self.mse_epochs, label="Treino", linewidth=2)
        plt.plot(self.val_mse_epochs, label="Validação", linewidth=2)
        plt.xlabel("Época")
        plt.ylabel("MSE")
        plt.title("Treino vs Validação")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_mse(self):
        plt.figure()
        plt.plot(self.mse_epochs, color="black")
        plt.title("Erro Quadrático Médio (MSE)")
        plt.xlabel("Época de treinamento")
        plt.ylabel("MSE")
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.8)
        plt.show()

    def plot_errors_by_folds(self):
        plt.figure()
        x = np.arange(1, len(self.fold_mse) + 1)
        plt.plot(x, self.fold_errors, marker='o')
        plt.xticks(x)
        plt.title("Erro por fold")
        plt.xlabel("Fold")
        plt.ylabel("Erro médio")
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.8)
        plt.show()

    def plot_mse_by_folds(self):
        plt.figure()
        x = np.arange(1, len(self.fold_mse) + 1)
        plt.plot(x, self.fold_mse, marker='o')
        plt.xticks(x)
        plt.title("MSE por fold")
        plt.xlabel("Fold")
        plt.ylabel("MSE médio")
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.8)
        plt.show()

    def plot_accuracy_by_folds(self):
        plt.figure()
        x = np.arange(1, len(self.fold_mse) + 1)
        plt.plot(x, self.fold_accuracies, marker='o')
        plt.xticks(x)
        plt.title("Acurácia por fold")
        plt.xlabel("Fold")
        plt.ylabel("Acurácia média")
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.8)
        plt.show()

    def plot_metrics_by_folds(self):
        import matplotlib.pyplot as plt
        import numpy as np
    
        x = np.arange(1, len(self.fold_accuracies) + 1)
    
        plt.figure()
        plt.plot(x, self.fold_accuracies, marker='o', label="Acurácia")
        plt.plot(x, self.fold_precision, marker='o', label="Precisão")
        plt.plot(x, self.fold_recall, marker='o', label="Recall")
        plt.plot(x, self.fold_f1, marker='o', label="F1-score")
    
        plt.xlabel("Fold")
        plt.ylabel("Valor")
        plt.title("Métricas por Fold")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
        plt.show()

    def plot_train_val_quartis(self):
        train = np.array(self.fold_mse_history)
        val = np.array(self.fold_val_mse_history)
    
        epochs = np.arange(train.shape[1])
    
        # estatísticas
        t_q1, t_med, t_q3 = np.percentile(train, [25,50,75], axis=0)
        v_q1, v_med, v_q3 = np.percentile(val, [25,50,75], axis=0)
    
        plt.figure()
    
        # treino
        plt.plot(epochs, t_med, label="Treino (mediana)", color='blue')
        plt.fill_between(epochs, t_q1, t_q3, alpha=0.2, color='blue')
    
        # validação
        plt.plot(epochs, v_med, label="Validação (mediana)", color='red', linestyle='--')
        plt.fill_between(epochs, v_q1, v_q3, alpha=0.2, color='red')
    
        plt.title("Treino vs Validação com faixa interquartil")
        plt.xlabel("Época")
        plt.ylabel("MSE")
    
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.8)
    
        plt.show()

    def plot_mse_by_folds_quartis(self):
        mse_matrix = np.array(self.fold_mse_history)  
        # shape: (n_folds, n_epochs)
    
        # estatísticas ao longo dos folds
        q1 = np.percentile(mse_matrix, 25, axis=0)
        median = np.percentile(mse_matrix, 50, axis=0)
        q3 = np.percentile(mse_matrix, 75, axis=0)
    
        epochs = np.arange(len(median))
    
        plt.figure()
    
        # mediana
        plt.plot(epochs, median, label="Mediana MSE", linewidth=2)
    
        # faixa interquartil
        plt.fill_between(epochs, q1, q3, alpha=0.2, label="Q1–Q3")
    
        plt.title("MSE por época (com quartis entre folds)")
        plt.xlabel("Época")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.8)
    
        plt.show()        
