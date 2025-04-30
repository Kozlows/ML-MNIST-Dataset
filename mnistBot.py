import numpy as np

class AI:
    def __init__(self):
        self.setup()  # set up the ai
        self.loss = []  # to store all the loss values

    def setup(self):  # Sets up all the layers of the AI
        self.layers = []  # Stores the layers here
        self.layers.append(ReLU(784, 256, first=True))  # LayerType(input amount, output amount, whether its first, default is False)
        self.layers.append(ReLU(256, 32))
        self.layers.append(SoftMax(32, 10))

    def crossEntropyDer(self, labels, results):  # calculates crossEntropy derivative of a batch
        epsilon = 1e-7  # Small value to prevent log(0)
        loss = -np.sum(labels * np.log(results + epsilon)) / labels.shape[0]  # the loss value
        self.loss.append(loss)  # stores the loss
        return results - labels  # this is the derivative of cross Entropy of the batch

    def train(self, inputs):  # trains ai based on inputs
        for batch in inputs:  # for every batch in inputs
            labels, results = self.run(batch, der=True)  # gets the labels from the batch, as well as the result of what happens when you run the pixels through all the layers
            der = self.crossEntropyDer(labels, results)  # calculates the corss entropy derivative of these labels and results
            for layer in self.layers[::-1]:  # goes through every layer from the oposite direction
                der = layer.backward(der)  # calculates the derivative of that layer, while updating that layers biases and weights


    def test(self, inputs):  # runs inputs over the layers, to test how accurate it is
        mean = 0
        total = 0
        for batch in inputs:  # for every batch in inputs
            labels, results = self.run(batch)  # get its labels and results of running the pixels through all the layers
            labels, results = np.argmax(labels, axis=1), np.argmax(results, axis=1)  # stores the index of hot-coded labels, stores the index of biggest result probability
            total += 1  # increment total to account for this batch
            mean += np.mean(labels == results)  # check for all labels which are correctly identified by results, get the mean correct accuracy for this batch, and add it to total mean
        return f"{mean/total * 100:.5f}%"  # calculate and return accuracy of all the batches together

    def run(self, batch, der=False):  # forward pass the batch through all the layers to obtain ai's guess to what number the pixels show
        labels = np.zeros((batch.shape[0], 10))  # initialises the labels to just 0's
        for i, line in enumerate(batch):  # for every line in the batch
            labels[i][int(line[0])] = 1  # hot code the specified index
        results = np.array([line[1:] for line in batch])  # seperate the label from the pixels, and store those pixels in results
        for layer in self.layers:  # itterate over every layer
            results = layer.forward(results, der)  # update the results based on forward pass in that layer
        return labels, results  # return hot coded labels, as well as the calcualted probabilities of each number based on pixels

class Layer:
    def __init__(self, inputs, ouputs, first=False):  # The amount of inputs this layer will take in, and the amount of outputs this layer will return, and whether this layer is first
        self.genNodes(inputs, ouputs)  # generates all the weights and biases of this layer
        self.rate = 0.01  # sets an initial learning rate
        self.decay = 0.008  # sets a decay, that will be used to simple decay the learning rate
        self.t = 0  # initialise time
        self.first=first  # define whether it is first or not
        self.setupAdam()  # set up all the variables needed for adam
    
    def updateRate(self):  # updates the learning rate based on time and set decay
        self.lr = self.rate / (1 + self.decay * self.t)  # this is simple decay

    def setupAdam(self):  # initialises all variables needed by adam
        self.mb = np.zeros(self.b.shape)  # momentum for bias nodes
        self.vb = np.zeros(self.b.shape)  # velocity for bias nodes
        self.mw = np.zeros(self.w.shape)  # momentum for weight nodes
        self.vw = np.zeros(self.w.shape)  # velocity for weight nodes
    
    def adam(self, bder, wder):  # adaptive momentum, updates the weights and biases of this layer
        self.updateRate()  # updates learning rate based on steps taken already
        beta1 = 0.9
        beta2 = 0.99
        e = 1e-8

        self.t += 1  # increment step/time
        self.mb = beta1 * self.mb + (1 - beta1) * bder  # momentum formula
        self.mw = beta1 * self.mw + (1 - beta1) * wder  # momentum formula
        self.vb = beta2 * self.vb + (1 - beta2) * (bder ** 2)  # velocity formula
        self.vw = beta2 * self.vw + (1 - beta2) * (wder ** 2)  # velocity formula

        mb_hat = self.mb / (1 - beta1 ** self.t)
        mw_hat = self.mw / (1 - beta1 ** self.t)
        vb_hat = self.vb / (1 - beta2 ** self.t)
        vw_hat = self.vw / (1 - beta2 ** self.t)

        self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + e)  # update biases based on adam
        self.w -= self.lr * mw_hat / (np.sqrt(vw_hat) + e)  # update weights based on adam

    def forward(self, input, der=False):  # forward pass of input
        computation = (input @ self.w) + self.b  # matrix multiply the input on the weights, before adding the bias onto it
        if der:  # if this is ran during backpropagation
            self.input = input  # update the stored input for later use to find derivative
        return self.activation(computation, der)  # return activated version of computation

    def backward(self, der):  # backward pass of derivative, for backpropagation
        biasDer = self.activationDer(der)  # gets the bias derivative based on input derivative and the activation layers derivative
        weightsDer = self.input.T @ biasDer / biasDer.shape[0]  # calculates the derivative of the weight, averaged over batch size
        biasDerAvg = np.mean(biasDer, axis=0)  # averages the bias derivative over batch size; why does the code break if the variable name is biasDer???
        self.adam(biasDerAvg, weightsDer)  # using adam, updates the weights and biases using their respective derivatives

        if not self.first:  # This just checks if its the first layer in the forward pass. Used to determine of the next derivative needs to be calculated or not
            der = biasDer @ self.w.T  # calculates this layers derivative
            return der  # passes it onto the next layer


    def genNodes(self, inputs, ouputs):  # abstract method of generating the bias and weights. This is defined in each subclasses of Layer
        return None

    def activation(self, input, der=False):  # abstract method of generating the activation function. This is defined in each subclasses of Layer
        return None
    
    def __str__(self):  # String version of a layer
        return f"Weights:\n{self.w}\nBiases:\n{self.b}"  # returns the layers weights and biases

class ReLU(Layer):  # ReLu hidden layer
    def activation(self, input, der=False):  # defines ReLu's activation function
        act = np.where(input > 0, input, 0)  # any value less then 0, makes it 0
        if der:  # if ran in backpropagation
            self.actD = np.where(input > 0, 1, 0)  # calculate the derivative of the activation based on input
        return act  # returns the result of running the input through the acitvation function

    def activationDer(self, der):  # calcualtes the activation functions derivative
        return der * self.actD

    def genNodes(self, inputs, outputs):  # He Initialisation
        self.w = np.random.normal(0, np.sqrt(2 / inputs), (inputs, outputs))
        self.b = np.random.normal(0, np.sqrt(2 / inputs), (outputs))

class SoftMax(Layer):  # softmax output layer
    def activation(self, input, der=False):  # defines softmax's activation function
        logits = np.exp(input - np.max(input, axis=1, keepdims=True))
        act = logits / np.sum(logits, axis=1, keepdims=True)
        return act

    def activationDer(self, der):
        return der  # returns input because its derivative is already calculated in the cross-entropy derivative, so nothing else needs to be calculated

    def genNodes(self, inputs, outputs):  # Xavier Initialisation
        self.w = np.random.randn(inputs, outputs) * np.sqrt(1 / (inputs + outputs))
        self.b = np.random.randn(outputs) * np.sqrt(1 / outputs)