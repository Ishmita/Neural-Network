# n = 4
# 2n = 8
# 3n = 12
def irisrun():
    import networkTry
    import random
    import IrisLoadData
    train , test = IrisLoadData.load()
    random.shuffle(train)
    test = train[125:150]
    train = train[0:125]
    net = networkTry.NeuralNetwork([4,4,3])
    net.SGD(train , 2000, 10, 0.1, test_data = test)
irisrun()
