import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 'Iris-setosa' -> 1
# 'Iris-versicolor' -> 2
# 'Iris-virginica' -> 3
def load():
    df = pd.read_csv('wine.csv')
    y = df.iloc[0:178, 0].values
    x = df.iloc[0:178, 1:14].values
    minx = []
    maxx = []
    mean = []
    for i in range(13):
        c = x[0:178, i]
        minx.append(np.amin(c))
        maxx.append(np.amax(c))
        mean.append(np.mean(c))
        norm = (c - mean[i])/(maxx[i]-minx[i])
        x[0:178, i] = norm
        print (norm)
    v = [np.zeros(3) for i in xrange(len(y))]
    for i in xrange(178):
        v[i][y[i]-1] = 1
    training_data = [(x1,y1) for x1,y1 in zip(x,v)]
    test_data = training_data[110:178]
    return (training_data, test_data)

load()    
