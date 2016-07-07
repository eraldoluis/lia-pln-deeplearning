#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from nnet.Layer import Layer


class DropoutLayer(Layer):
    def __init__(self, _input, noiseRate, seed):
        super(DropoutLayer, self).__init__(_input)
        theano_rng = RandomStreams(seed)

        inp = self.getInput()
        self.__output = theano_rng.binomial(size=inp.shape, n=1,
                                          p=1 - noiseRate,
                                          dtype=theano.config.floatX) * inp

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
