#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano.tensor as T

from NNet.Layer import Layer


class FlattenLayer(Layer):

    def __init__(self,_input):
        super(FlattenLayer, self).__init__(_input)

        # We flat to two dimension, because of the mini-batch
        self.__output = T.flatten(self.getInput(), 2)

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
