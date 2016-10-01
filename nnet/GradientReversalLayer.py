from ext_theano.GradientReversalOp import GradientReversalOp
from nnet.Layer import Layer


class GradientReversalLayer(Layer):
    def __init__(self, _input, _lambda):
        super(GradientReversalLayer, self).__init__(_input)


        r = GradientReversalOp(_lambda)
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
