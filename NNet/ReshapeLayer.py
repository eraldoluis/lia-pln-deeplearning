#!/usr/bin/env python
# -*- coding: utf-8 -*-
from NNet.Layer import Layer
import theano.tensor as T


class ReshapeLayer(Layer):
    def __init__(self, _input, shape):
        super(ReshapeLayer, self).__init__(_input)

        self.__output = T.reshape(self.getInput(), shape)

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
