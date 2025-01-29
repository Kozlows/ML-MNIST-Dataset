import numpy as np

class AI:
    def __init__(self):
        self.setup()

    def setup(self):  # Layer(input amount, output amount)
        self.layers = []
        self.layers.append(ReLU(784, 10))

    def train(self, inputs):
        pass

    def test(self, inputs):
        for batch in inputs:
            self.run(batch)

    def run(self, batch):
        labels = np.array([line[0] for line in batch])
        result = np.array([line[1:] for line in batch])
        for layer in self.layers:
            result = layer.forward(result)
        #print(result)

class Layer:
    def __init__(self, inputs, ouputs):  # The amount of inputs this layer will take in, and the amount of outputs this layer will return
        self.genNodes(inputs, ouputs)

    def forward(self, input, der=False):
        print(f"Input Shape: {input.shape}\nWeights Shape: {self.w.shape}\nBias Shape: {self.b.shape}")
        computation = (input @ self.w) + self.b
        print(f"Computation Shape: {computation.shape}")
        return self.activation(computation, der)

    def backward(self, input):
        pass

    def genNodes(self, inputs, ouputs):
        return None

    def activation(self, input, der=False):
        return input
    
    def __str__(self):
        return f"Weights:\n{self.w}\nBiases:\n{self.b}"

class ReLU(Layer):
    def activation(self, input, der=False):
        act = np.where(input > 0, input, 0)
        if der:
            self.actD = np.where(input > 0, 1, 0)
        return act

    def genNodes(self, inputs, outputs):  # He Initialisation
        self.w = np.random.normal(0, np.sqrt(2 / inputs), (inputs, outputs))
        self.b = np.random.normal(0, np.sqrt(2 / inputs), (outputs))

class SoftMax(Layer):
    def activation(self, input, der=False):
        logits = np.exp(input - np.max(input, axis=1, keepdims=True))
        act = logits / np.sum(logits, axis=1, keepdims=True)

        if der:
            n, m = act.shape
            self.actD = np.zeros(n, m, m)
            for i, batch in enumerate(act):
                jacobian = np.diag(batch) - np.outer(batch, batch)
                self.actD[i] = jacobian

        return act

    def genNodes(self, outputs, inputs):  # Xavier Initialisation
        self.w = np.random.normal(0, np.sqrt(1 / inputs), (inputs, outputs))
        self.b = np.random.normal(0, np.sqrt(1 / inputs), (outputs))