def irisrun():
    import networkTry
    import IrisLoadData
    train , test = IrisLoadData.load()
    net = networkTry.NeuralNetwork([4,5,3])
    net.SGD(train , 100, 10, 0.5, test_data = test)
