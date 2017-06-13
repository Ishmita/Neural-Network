import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
# 'Iris-setosa' -> 1
# 'Iris-versicolor' -> 2
# 'Iris-virginica' -> 3
"""df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
x = df.iloc[0:150, 0:4].values
y = np.where(y == 'Iris-setosa', 1, 2)
z = df.iloc[100:150, 4].values
z = np.where(z == 'Iris-virginica', 3, 1)
y = np.concatenate((y,z))
plt.scatter(x[:50, 0:2], x[:50, 2:4], color='red', marker='o', label='setosa')
plt.scatter(x[50:100, 0:2], x[50:100, 2:4], color='blue', marker='x', label='versicolor')
plt.scatter(x[100:150, 0:2], x[100:150, 2:4], color='green', marker='x', label='versicolor')
plt.xlabel('sepal')
plt.ylabel('petal')
plt.show()"""

                                       
class NeuralNetwork(object):

    def __init__(self, size):
        np.random.seed(0)
        self.num_layers = len(size)
        self.size = size
        self.biases = [np.random.randn(y) for y in size[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(size[:-1], size[1:])]

    def step_func(x):
        return 0 if x <= 0 else 1

    def feed_forward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        
        if test_data: n_test = len(test_data)
        n = len(training_data)
        r = 0.5
        for j in range(epochs):
            random.shuffle(training_data, lambda: r)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2][:,None])
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1][:, None])
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        count = 0
        for x,y in test_data:
            a = self.feed_forward(x)
            #print ("predicted: "+ str(a) + " actual: "+ str(y))
            pindex = np.argmax(a)
            aindex = np.where(y == 1)
            if pindex == aindex:
                count = count +1
        return count
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
            

        
