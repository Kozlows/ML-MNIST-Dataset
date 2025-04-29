import mnistBot as Bot
import numpy as np

tests = np.load("test.npy")

training = np.load("train.npy")


#print(tests.shape, training.shape)

batchSize = 10
tests = tests.reshape(-1, batchSize, 785)
training = training.reshape(-1, batchSize, 785)

#print(tests.shape, training.shape)

bot = Bot.AI()

before = bot.test(tests)

bot.train(training)
after = bot.test(tests)
#print(bot.loss)
print(f"Before Training: {before}\nAfter Training: {after}")
#print(f"Min, i: {np.min(bot.loss)} {np.argmin(bot.loss)}\nMax, i: {np.max(bot.loss)} {np.argmax(bot.loss)}")
#print(f"Size: {len(bot.loss)}")