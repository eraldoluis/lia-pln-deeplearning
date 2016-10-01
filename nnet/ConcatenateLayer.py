#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano.tensor as T
from nnet.Layer import Layer


class ConcatenateLayer(Layer):
    def __init__(self, _input, axis=1):
        super(ConcatenateLayer, self).__init__(_input)

        self.__output = T.concatenate(self.getInput(), axis=axis)

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
