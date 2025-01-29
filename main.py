import mnistBot as Bot
import numpy as np

testPath = "mnist_test.csv"
with open(testPath, "r") as f:
    tests = np.array([[int(n) if i == 0 else int(n) / 255 for i, n in enumerate(line.strip().split(","))] for line in f.readlines()])

batchSize = 10
tests = tests.reshape(-1, batchSize, 785)

bot = Bot.AI()

bot.test(tests[:2])