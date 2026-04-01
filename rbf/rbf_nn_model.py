import numpy as np

class RBFNeuron:
    def __init__(self, input_size, sigma):
        random_gen = np.random.default_rng()
        self.center = random_gen.uniform(0, 1, size=(input_size,1))
        self.center = self.center/np.linalg.norm(self.center) #center must have norm equal to 1
        self.sigma = sigma

    def output(self, x):
        x_diff_center_n  = np.linalg.norm(x-self.center)
        return np.exp( - (x_diff_center_n**2)/ (2*self.sigma**2))


class RBFNeuralNetwork:
    def __init__(self, n_inputs, n_neurons, n_outputs, sigma, output_fcn, d_output_fcn):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons + 1 # to include bias neuron
        self.n_outputs = n_outputs
        self.output_fcn = output_fcn
        self.output_fcn_map = np.vectorize(self.output_fcn)
        self.d_output_fcn = d_output_fcn
        self.d_output_fcn_map = np.vectorize(self.d_output_fcn)
        self.neurons = []
        self.neurons_output = np.zeros( (self.n_neurons, 1) )
        self.network_v = np.zeros((n_outputs,1))
        self.network_output = np.zeros( (n_outputs, 1) )
        for i in range(self.n_neurons -1 ): #individual neuron initialization, except bias neuron
            self.neurons.append(RBFNeuron(n_inputs, sigma))
        
        self.weights = np.random.randn(n_outputs, self.n_neurons ) #output matrix initialization
    def output(self, x):
        for i in range(self.n_neurons):
            if i < ( self.n_neurons - 1) :
                self.neurons_output[i,0] = self.neurons[i].output(x)
            else:
                self.neurons_output[i,0] = 1 #bias neuron
        self.network_v = self.weights @ self.neurons_output
        self.network_output = self.output_fcn_map(self.network_v)
        return self.network_output

    def learn(self, x, y_d, eta):
        y_e = self.output(x)
        e = y_d - y_e
        #delta = e * self.d_output_fcn_map(self.network_v) #for mse
        delta = e # for cross-entropy
        self.weights = self.weights + eta *  (delta  @ self.neurons_output.T)
        return e

        
        
        
        

    