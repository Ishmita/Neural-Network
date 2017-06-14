# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

z = [-1,-6,-4,-2,0,0.5,1,2,3,4,5,6,8]
min = -2
max = 3

def plot(a,z):
    print (a)
    plt.scatter(z, a, color='red')
    plt.xlabel('z')
    plt.ylabel('@(z)')
    plt.show()

def identity_func(z):
    plt.scatter(z, z, color='red')
    plt.xlabel('z')
    plt.ylabel('@(z)')
    plt.show()
    return z

identity_func(z)

def step_func(z):
    a = [] 
    for e in z:
        a.append(0 if e <= 0 else 1)
    return a


a = step_func(z)
plot(a,z)

def piecewise_linear(z):
    a = []
    for e in z:
        if e < min:
            a.append(0)
        elif e > max:
            a.append(1)
        else:
            m = 1.0/(max-min)
            b = 1-m*max
            a.append(m*e + b)
    return a

a = piecewise_linear(z)
plot(a,z)

def sigmoid(z):
    a = []
    for e in z:
        a.append(1.0/(1+np.exp(-e)))
    return a

a = sigmoid(z)
plot(a,z)

def complementary_log_log(z):
    a = []
    for e in z:
        a.append(1.0 - (np.exp(-(np.exp(e)))))

    return a

a = complementary_log_log(z)
plot(a,z)

def bipolar(z):
    a = []
    for e in z:
        if e <= 0:
            a.append(-1)
        else:
            a.append(1)
    return a

a = bipolar(z)
plot(a,z)

def bipolar_sigmoid(z):
    a = []
    for e in z:
        n = 1.0 - (np.exp(-e))
        d = 1.0 + (np.exp(-e))
        a.append(n/d)
    return a

a = bipolar_sigmoid(z)
plot(a,z)

def tanh(z):
    return np.tanh(z)

a = tanh(z)
plot(a,z)

#not working
def leCun_tanh(z):
    x = []
    for e in z:
        x.append((2/3)*e)
    z = x
    a = np.tanh(z)
    x = []
    for e in a:
        x.append(1.7159*e)
    return x

a = leCun_tanh(z)
plot(a,z)
