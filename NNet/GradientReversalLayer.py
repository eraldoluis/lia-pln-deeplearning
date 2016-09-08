from NNet.Layer import Layer
from ext_theano.ReverseGradient import ReverseGradient


class GradientReversalLayer(Layer):
    def __init__(self, _input, _lambda):
        super(GradientReversalLayer, self).__init__(_input)


        r = ReverseGradient(_lambda)
        self.__output = r(self.getInput())

    def getOutput(self):
        return self.__output

    def getParameters(self):
        return []

    def getDefaultGradParameters(self):
        return []

    def getStructuredParameters(self):
        return []

    def getUpdates(self, cost, lr, sumSqGrads=None):
        return []
