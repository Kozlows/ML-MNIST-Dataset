import numpy as np

class AI:
    def __init__(self):
        self.setup()
        self.loss = []

    def setup(self):  # Layer(input amount, output amount)
        self.layers = []
        self.layers.append(ReLU(784, 128, first=True))
        self.layers.append(ReLU(128, 16))
        self.layers.append(SoftMax(16, 10))
        
        #self.layers.append(SoftMax(784, 10, first=True))
        #self.layers.append(ReLU(784, 10, first=True))
        #self.layers.append(Sigmoid(784, 10, first=True))


    def MSEDer(self, labels, results):
        loss = 1/2 * np.mean(np.power(labels - results, 2))
        self.loss.append(loss)
        return labels - results

    def crossEntropyDer(self, labels, results):
        epsilon = 1e-7  # Small value to prevent log(0)
        loss = -np.sum(labels * np.log(results + epsilon)) / labels.shape[0]
        self.loss.append(loss)
        return results - labels
        #print(der)

    def train(self, inputs):  # To confirm if works
        for batch in inputs:
            labels, results = self.run(batch, der=True)
            #der = self.crossEntropyDer(labels, results)
            #print(der.shape)
            #der = self.MSEDer(labels, results)
            #print(der.shape)
            der = self.crossEntropyDer(labels, results)
            #print(der)
            #print(f"Labels: {labels.shape}\nResults: {results.shape}")
            #print(der)
            for layer in self.layers[::-1]:
                der = layer.backward(der)


    def test(self, inputs):
        mean = 0
        total = 0
        for batch in inputs:
            labels, results = self.run(batch)
            labels, results = np.argmax(labels, axis=1), np.argmax(results, axis=1)
            total += 1
            mean += np.mean(labels == results)
        return f"{mean/total * 100:.5f}%"




    def run(self, batch, der=False):
        labels = np.zeros((batch.shape[0], 10))
        for i, line in enumerate(batch):
            labels[i][int(line[0])] = 1
        results = np.array([line[1:] for line in batch])
        #print(np.average(results), np.sum(results))
        for layer in self.layers:
            results = layer.forward(results, der)
        return labels, results

class Layer:
    def __init__(self, inputs, ouputs, first=False):  # The amount of inputs this layer will take in, and the amount of outputs this layer will return
        self.genNodes(inputs, ouputs)
        self.rate = 0.008  # To further modify so its not static
        self.first=first
        self.setupAdam()
    
    def setupAdam(self):
        self.mb = np.zeros(self.b.shape)
        self.vb = np.zeros(self.b.shape)
        self.mw = np.zeros(self.w.shape)
        self.vw = np.zeros(self.w.shape)
        self.thetaB = np.ones(self.b.shape) * self.rate
        self.thetaW = np.ones(self.w.shape) * self.rate
        self.t = 0
        self.beta1 = 0
        self.beta2 = 0
        self.e = 1e-8
    
    def updateTheta(self, bder, wder):
        bder = np.mean(bder, axis=0)
        self.t += 1
        self.mb = np.nan_to_num(self.beta1 * self.mb + (1 - self.beta1) * bder, nan=0.0)
        self.mw = np.nan_to_num(self.beta1 * self.mw + (1 - self.beta1) * wder, nan=0.0)
        self.vb = np.nan_to_num(self.beta2 * self.vb + (1 - self.beta2) * (bder ** 2), nan=0.0)
        self.vw = (self.beta2 * self.vw) + ((1 - self.beta2) * (wder ** 2))
        #print(np.max(self.vw), np.min(self.vw))
        #print(self.vw)
        #input("Press Enter...")
        #if(np.min(self.vw) <= 0):
        #    print(wder)
        #    exit()
        mb_hat = self.mb / (1 - self.beta1 ** self.t)
        mw_hat = self.mw / (1 - self.beta1 ** self.t)
        vb_hat = self.vb / (1 - self.beta2 ** self.t)
        vw_hat = self.vw / (1 - self.beta2 ** self.t)

        self.thetaB = self.rate * mb_hat / (np.sqrt(vb_hat) + self.e)
        self.thetaW = self.rate * mw_hat / (np.sqrt(vw_hat) + self.e)

        #print(f"B:\n{thetaB.shape}")
        #print(f"W:\n{thetaW.shape}")

    def forward(self, input, der=False):
        #print(input.shape, self.w.shape, self.b.shape)
        computation = (input @ self.w) + self.b
        if der:
            self.input = input
        #print(computation)
        return self.activation(computation, der)

    def backward(self, der):  # Input shapes are 100% correct;
        biasDer = self.activationDer(der)  # Calculates the bias Derivative by multiplying derivative from previous layer/cost function
                                           # With the derivative of the activation function
        weightsDer = self.input.T @ biasDer / biasDer.shape[0]
        #self.updateTheta(biasDer, weightsDer)
        self.b -= self.rate * np.mean(biasDer, axis=0)  #np.sum(biasDer, axis=0)  # the rate is defined earlier in my code as self.rate = 0.000001
        #print(self.input.T.shape, biasDer.shape, self.w.shape)
        
        #print(weightsDer.shape)
        self.w -= self.rate * weightsDer #np.sum(weightsDer, axis=0)
        #print(biasDer.shape, self.b.shape, self.input.T.shape, weightsDer.shape, self.w.shape)

        if not self.first:  # This just checks if its the first layer in the forward pass
            # Because if it is, it does not need to return any derivative
            der = biasDer @ self.w.T
            return der


    def genNodes(self, inputs, ouputs):
        return None

    def activation(self, input, der=False):
        return None
    
    def __str__(self):
        return f"Weights:\n{self.w}\nBiases:\n{self.b}"

class ReLU(Layer):
    def activation(self, input, der=False):
        act = np.where(input > 0, input, 0)
        if der:
            self.actD = np.where(input > 0, 1, 0)
        return act

    def activationDer(self, der):
        return der * self.actD

    def genNodes(self, inputs, outputs):  # He Initialisation
        #self.w = np.random.random((inputs, outputs))
        #self.b = np.random.random(outputs)
        self.w = np.random.normal(0, np.sqrt(2 / inputs), (inputs, outputs))
        self.b = np.random.normal(0, np.sqrt(2 / inputs), (outputs))

class SoftMax(Layer):
    def activation(self, input, der=False):
        logits = np.exp(input - np.max(input, axis=1, keepdims=True))
        act = logits / np.sum(logits, axis=1, keepdims=True)
        return act

    def activationDer(self, der):
        return der 

    def genNodes(self, inputs, outputs):  # Xavier Initialisation
        #self.w = np.random.random((inputs, outputs))
        #self.b = np.random.random(outputs)

        self.w = np.random.randn(inputs, outputs) * np.sqrt(1 / (inputs + outputs))
        self.b = np.random.randn(outputs) * np.sqrt(1 / outputs)


class Sigmoid(Layer):
    def activation(self, input, der=False):
        act = np.power(1 + np.exp(-input), -1)
        if der:
            self.act = act

        return act

    def activationDer(self, der):
        return self.act * (1 - self.act)

    def genNodes(self, inputs, outputs):  # Xavier Initialisation
        self.w = np.random.randn(inputs, outputs) * np.sqrt(1 / (inputs + outputs))
        self.b = np.random.randn(outputs) * np.sqrt(1 / outputs)