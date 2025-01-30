import mnistBot as Bot
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


batchSize = 10
tests = tests.reshape(-1, batchSize, 785)
training = training.reshape(-1, batchSize, 785)
print(training.shape)
bot = Bot.AI()

bot.train(training)
bot.test(tests)