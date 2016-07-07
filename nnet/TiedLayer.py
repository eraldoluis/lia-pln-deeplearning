#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import theano
import theano.tensor as T
from theano.tensor.sharedvar import TensorSharedVariable

from nnet import LinearLayer
from nnet.Layer import Layer


class TiedLayer(Layer):
    '''
    This represents a layer which your W is equal a transpose of W from the previous layer.
    '''

    def __init__(self, _input, W, lenOut, b=None):
        """
        :type
        :param _input: previous layer

        :param W: parameter from the previous layer which will be tranposed.

        :type lenOut: int
        :param lenOut: number of units
        """
        super(TiedLayer, self).__init__(_input)

        if not isinstance(b, TensorSharedVariable):
            if isinstance(b, (numpy.ndarray, list)):
                b_values = numpy.asarray(b, dtype=theano.config.floatX)
            else:
                b_values = numpy.zeros(lenOut, dtype=theano.config.floatX)

            b = theano.shared(value=b_values, name='b_hiddenLayer', borrow=True)

        self.W = W.T
        self.b = b

        self.__output = T.dot(self.getInput(), self.W) + self.b

        # parameters of the model
        self.params = [self.b]

    def getOutput(self):
        return self.__output

    def getParameters(self):
        return self.params

    def getDefaultGradParameters(self):
        return self.params

    def getStructuredParameters(self):
        return []

    def getUpdates(self, cost, lr, sumSqGrads=None):
        return []


