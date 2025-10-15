import numpy as np

class Layer:
    """
    A single layer in a neural network.

    
    weights: Weights matrix for the layer.
    biases: Biases vector for the layer.
    inputs: Inputs to the layer from the previous layer.
    u_minus_g_threshold: Threshold for gradient adjustment during backpropagation.
    """

    def __init__(self, in_size, out_size, u_minus_g_threshold=None):
        """
        runs the layer with random weights and zero biases.

        
        in_size: Number of input neurons.
        out_size: Number of output neurons.
        u_minus_g_threshold: Threshold for gradient adjustment.
        """
        self.weights = np.random.randn(in_size, out_size) * np.sqrt(2. / in_size)
        self.biases = np.zeros(out_size)
        self.inputs = None
        self.u_minus_g_threshold = u_minus_g_threshold

    def forward(self, x):
        """
        the forward pass through the layer.
        x : Input data.
        Returns output of the layer.
        """
        self.inputs = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, dC_da):
        """
        The backward pass through the layer.
        dC_da: Gradient of the cost
        Returns gradients with respect to the inputs, biases, and weights.
        """
        dC_db = dC_da
        dC_dw = np.dot(self.inputs.T, dC_da)
        if self.u_minus_g_threshold is not None:
            dC_dw[self.inputs < self.u_minus_g_threshold] *= self.u_minus_g_threshold
        return dC_da.dot(self.weights.T), dC_db, dC_dw

def sigmoid(z):
    """
    Computes the sigmoid function.
    z: Input value or array.

    Returnsthe sigmoid of the input.
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):

    return sigmoid(z) * (1 - sigmoid(z))

def pairwise(iterable):
    """
    makes pairs of successive elements from the input iterable.

    iterable : Input iterable.

    Returns the iterator of pairs of successive elements.
    """
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class NeuralNetwork:
    """
    A neural network

    sizes: List of layer sizes.
    layers: List of Layer objects.
    u_minus_g_threshold: Threshold for gradient adjustment.
    num_layers : Number of layers in the network.
    """

    def __init__(self, sizes, u_minus_g_threshold):
        """
        sets up the neural network.

        sizes: List of layer sizes.
        u_minus_g_threshold : Threshold for gradient adjustment.
        """
        self.sizes = sizes
        self.layers = []
        self.u_minus_g_threshold = u_minus_g_threshold
        for in_size, out_size in pairwise(sizes):
            self.layers.append(Layer(in_size, out_size, u_minus_g_threshold))
        self.num_layers = len(self.layers)

    def __repr__(self):
        return f"NeuralNetwork({self.sizes})"

    def feedforward(self, x):
        """
        Performs the feedforward pass through the network.
        """
        for layer in self.layers[:-1]:
            x = np.tanh(layer.forward(x))
        return sigmoid(self.layers[-1].forward(x))

    def evaluate(self, Xtest, ytest):
        """
        Evaluates the network's performance on the test set.
        """
        test_results = np.array([1 if self.feedforward(x) > 0.5 else 0 for x in Xtest])
        return np.sum(test_results == ytest)

    def cost(self, Xtest, ytest):
        """
        Computes the cost function on the test set.
        """
        outputs = np.array([self.feedforward(x) for x in Xtest]).squeeze()
        n = len(ytest)
        return -np.sum(ytest * np.log(outputs) + (1 - ytest) * np.log(1 - outputs)) / n

    def SGD(self, Xtrain, ytrain, Xtest, ytest, epochs=4, eta=0.1):
        """
        Trains the network using stochastic gradient descent.
        epochs : Number of epochs
        eta : Learning rate
        """
        self.train_scores = []
        self.test_scores = []
        for iepoch in range(epochs):
            for x, y in zip(Xtrain, ytrain):
                nabla_b, nabla_w = self.backprop(x, y)
                for l in range(self.num_layers):
                    self.layers[l].weights -= eta * nabla_w[l]
                    self.layers[l].biases -= eta * nabla_b[l]

            self.train_scores.append(self.evaluate(Xtrain, ytrain) / len(ytrain))
            self.test_scores.append(self.evaluate(Xtest, ytest) / len(ytest))

            if iepoch % 10 == 0:
                print(f"Epoch #{iepoch}")
                print(f"Performance: {self.evaluate(Xtest, ytest)} / {len(ytest)}")

    def backprop(self, x, y):
        """
        Performs the backpropagation algorithm to compute gradients.
        """
        activations = [x]
        zs = []
        activation = x
        for layer in self.layers[:-1]:
            z = layer.forward(activation)
            zs.append(z)
            activation = np.tanh(z)
            activations.append(activation)

        z = self.layers[-1].forward(activation)
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

        delta = activation - y
        nabla_b = [delta]
        nabla_w = [np.dot(activations[-2].reshape(-1, 1), delta.reshape(1, -1))]

        for l in range(2, self.num_layers + 1):
            z = zs[-l]
            sp = sigmoid_derivative(z) if l == 2 else 1 - np.tanh(z)**2
            delta = np.dot(delta, self.layers[-l + 1].weights.T) * sp
            nabla_b.append(delta)
            nabla_w.append(np.dot(activations[-l - 1].reshape(-1, 1), delta.reshape(1, -1)))

        return nabla_b[::-1], nabla_w[::-1]

    def predict(self, X):
        """
        Predicts the labels for the given input data.
        """
        predictions = []
        for x in X:
            prediction = self.feedforward(x)
            predictions.append(1 if prediction > 0.5 else 0)
        return np.array(predictions)