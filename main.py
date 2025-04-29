import mnistBot as Bot
import numpy as np

testPath = "mnist_test.csv"
with open(testPath, "r") as f:
    tests = np.array([[int(n) if i == 0 else int(n) / 255 for i, n in enumerate(line.strip().split(","))] for line in f.readlines()[:8000]])

training = []
for i in range(2):
    trainPath = f"mnist_train{i}.csv"
    with open(trainPath, "r") as f:
        training.extend([[int(n) if i == 0 else int(n) / 255 for i, n in enumerate(line.strip().split(","))] for line in f.readlines()[:4000]])
training = np.array(training)


batchSize = 1
tests = tests.reshape(-1, batchSize, 785)
training = training.reshape(-1, batchSize, 785)
bot = Bot.AI()

before = bot.test(tests)

bot.train(training)
after = bot.test(tests)
print(bot.loss)
print(f"Before Training: {before}\nAfter Training: {after}")
print(f"Min, i: {np.min(bot.loss)} {np.argmin(bot.loss)}\nMax, i: {np.max(bot.loss)} {np.argmax(bot.loss)}")
print(f"Size: {len(bot.loss)}")