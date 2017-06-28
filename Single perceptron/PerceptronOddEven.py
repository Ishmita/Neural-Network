# Perceptron.py
from numpy import random, array, dot
from random import choice
import matplotlib.pyplot as plt
# even -> 0, odd ->1
training_data = (([1,3, 0.5], 0), ([4,5, 0.5], 1), ([2,7, 0.5], 1), ([3,2, 0.5], 1),
                 ([5,1, 0.5], 0), ([8,3, 0.5], 1), ([7,5, 0.5], 0), ([4,6, 0.5], 0),
                 ([4,3, 0.5], 1), ([12,10, 0.5], 0), ([14,4, 0.5], 0), ([17,2, 0.5], 1),
                 ([13,4, 0.5], 1), ([19,1, 0.5], 0), ([9,2, 0.5], 1), ([11,1, 0.5], 0))

w = random.rand(3)

def step_func(x):
	return 0 if x < 0 else 1

n = 100
eta = 0.1
errors = []	


for i in range(n):
	x , expected = choice(training_data)
	result = dot(w, x)
	error = expected - step_func(result)
	errors.append(error)
	x = [error * eta * i for i in x]
	w = [i + j for i,j in zip(x,w)]
	
for x,y in training_data:
	result = dot(w , x)
	print ("{}: {} -> {}".format(x[:2], result, step_func(result)))

#plt.plot([i for i in range(n)], errors)
#plt.show()
#plt.ylabel('error')
#plt.xlabel('epoch')
