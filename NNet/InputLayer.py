#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NNet.Layer import Layer
from util.util import getTheanoTypeByDimension


class InputLayer(Layer):
    def __init__(self, ndim, name=None, dtype=None):
        '''
        Represents an input layer of a neural network.

        :type ndim: int
        :param ndim: number of dimensions.
            Always this value will be incremented with 1, because of the mini-batch.

        :param name: input name

        :param dtype: The types defined by theano.
        '''
        d = ndim + 1

        _input = getTheanoTypeByDimension(d, name, dtype)
        super(InputLayer, self).__init__(_input, False)

    def getOutput(self):
        return self.getInputs()[0]

    def getParameters(self):
        return []

    def getDefaultGradParameters(self):
        return []

    def getStructuredParameters(self):
        return []

    def getUpdates(self, cost, lr, sumSqGrads=None):
        return []
