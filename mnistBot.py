import numpy as np

class AI:
    def __init__(self):
        self.setup()

    def setup(self):  # Layer(input amount, output amount)
        self.layers = []
        self.layers.append(ReLU(784, 10))

    def train(self, inputs):  # To confirm if works
        for batch in inputs:
            labels, results = self.run(batch, der=True)
            # cost = 1/2 * (labels - results) ** 2
            der = labels - results
            #print(f"Labels: {labels.shape}\nResults: {results.shape}")
            #print(der)
            for layer in self.layers[::-1]:
                der = layer.backward(der)


    def test(self, inputs):
        for batch in inputs:
            self.run(batch)

    def run(self, batch, der=False):
        labels = np.array([[0 if i != int(line[0]) else 1 for i in range(10)] for line in batch])
        results = np.array([line[1:] for line in batch])
        for layer in self.layers:
            results = layer.forward(results, der)
        return labels, results

class Layer:
    def __init__(self, inputs, ouputs):  # The amount of inputs this layer will take in, and the amount of outputs this layer will return
        self.genNodes(inputs, ouputs)
        self.rate = 0.05  # To further modify so its not static

    def forward(self, input, der=False):
        #print(f"Input Shape: {input.shape}\nWeights Shape: {self.w.shape}\nBias Shape: {self.b.shape}")
        computation = (input @ self.w) + self.b
        #print(f"Computation Shape: {computation.shape}")
        if der:
            self.input = input
        return self.activation(computation, der)

    def backward(self, der):  # Many problems here, to be fixed later
        biasDer = der * self.actD  # Not too confident in this, especially in softmax since the shape will be different there by an entire dimension :p
        print(biasDer.shape, der.shape, self.actD.shape, np.average(biasDer, axis=0).shape)
        self.b -= self.rate * np.average(biasDer, axis=0)  # Not confident in this either
        print(biasDer.shape, self.input.shape, self.w.shape)
        weightsDer = self.input.T @ biasDer  # I kinda just assumed this based on nothing but their shapes, will have to look into it in more detail later
        self.w -= self.rate * np.average(weightsDer, axis=0)  # I'm honestly just guessing at this point, more research required
        print(biasDer.shape, self.w.shape)
        return biasDer
        #  return biasDer * self.w, except it somehow needs to make sense


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