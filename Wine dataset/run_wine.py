# n = 13
# 2n = 26
# 3n = 39
def wine_run():
    import networkTry
    import random
    import load_wine_data as data
    train , test = data.load()
    random.shuffle(train)
    test = train[138:178]
    train = train[0:138]
    net = networkTry.NeuralNetwork([13,39,3])
    net.SGD(train , 2000, 10, 0.1, test_data = test)
wine_run()
