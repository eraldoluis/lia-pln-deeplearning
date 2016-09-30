#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano.tensor as T

from nnet.Layer import Layer


class ReshapeLayer(Layer):

    def __init__(self, _input, newShape, nDim=None):
        super(ReshapeLayer, self).__init__(_input)
        # Reshape input.
        self.__output = T.reshape(self.getInput(), newShape, nDim)

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
