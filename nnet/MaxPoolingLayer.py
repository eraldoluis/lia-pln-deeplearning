#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano.tensor as T

from nnet.Layer import Layer


class MaxPoolingLayer(Layer):

    def __init__(self,_input):
        super(MaxPoolingLayer, self).__init__(_input, trainable=False)

        # We apply a max operator along the first dimension.
        self.__output = T.max(self.getInput(), axis=0)

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
