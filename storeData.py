import json
import numpy as np

testPath = "mnist_test.csv"
with open(testPath, "r") as f:
    tests = np.array([[int(n) if i == 0 else int(n) / 255 for i, n in enumerate(line.strip().split(","))] for line in f.readlines()])

training = []
for i in range(2):
    trainPath = f"mnist_train{i}.csv"
    with open(trainPath, "r") as f:
        training.extend([[int(n) if i == 0 else int(n) / 255 for i, n in enumerate(line.strip().split(","))] for line in f.readlines()])
training = np.array(training)

np.save("test.npy", tests)
np.save("train.npy", training)