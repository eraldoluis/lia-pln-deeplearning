#!/usr/bin/env python
import theano.tensor as T

# Activations functions
from NNet.Layer import Layer


def softmax(x):
    return T.nnet.softmax(x)


def softplus(x):
    return T.nnet.softplus(x)


def tanh(x):
    return T.tanh(x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)


# Activation Layer

class ActivationLayer(Layer):

    def __init__(self,_input, actFunction):
        """
        :param actFunction: activation function
        """
        super(ActivationLayer, self).__init__(_input)

        self.__output =  actFunction(self.getInput())


    def getOutput(self):
        return self.__output

    def getParameters(self):
        return []

    def getStructuredParameters(self):
        return []

    def getDefaultGradParameters(self):
        return []

    def getUpdates(self, cost, learningRate, sumSqGrads=None):
        return []




