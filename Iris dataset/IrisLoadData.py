import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 'Iris-setosa' -> 1
# 'Iris-versicolor' -> 2
# 'Iris-virginica' -> 3
def load():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    x = df.iloc[0:150, 0:4].values
    y = np.where(y == 'Iris-setosa', 1, 2)
    z = df.iloc[100:150, 4].values
    z = np.where(z == 'Iris-virginica', 3, 1)
    y = np.concatenate((y,z))
    v = [np.zeros(3) for i in xrange(len(y))]
    for i in xrange(150):
        v[i][y[i]-1] = 1
    training_data = [(x1,y1) for x1,y1 in zip(x,v)]
    test_data = training_data[25:125]
    return (training_data, test_data)

load()    
"""plt.scatter(x[:50, 0:2], x[:50, 2:4], color='red', marker='o', label='setosa')
plt.scatter(x[50:100, 0:2], x[50:100, 2:4], color='blue', marker='x', label='versicolor')
plt.scatter(x[100:150, 0:2], x[100:150, 2:4], color='green', marker='x', label='versicolor')
plt.xlabel('sepal')
plt.ylabel('petal')
plt.show()"""
