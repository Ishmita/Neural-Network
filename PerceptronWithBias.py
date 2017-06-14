# Perceptron.py
from numpy import random, array, dot
from random import choice
import matplotlib.pyplot as plt
# 0 -> even, 1 -> odd
training_data = (([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1))
bias = random.rand(1)
w = random.rand(2)

def step_func(x):
	return 0 if x < 0 else 1

n = 200
eta = 0.1
errors = []	


for i in range(n):
	x , expected = choice(training_data)
	result = dot(w, x) + bias
	error = expected - step_func(result)
	errors.append(error)
	x = [error * eta * i for i in x]
	w = [i + j for i,j in zip(x,w)]
	bias = bias + error * eta
	
	
for x,y in training_data:
	result = dot(w , x)
	print ("{}: {} -> {}".format(x[:2], result, step_func(result)))

plt.plot([i for i in range(n)], errors)
plt.show()
plt.ylabel('error')
plt.xlabel('epoch')
