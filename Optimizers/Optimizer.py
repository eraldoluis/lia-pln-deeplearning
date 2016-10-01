#!/usr/bin/env python
# -*- coding: utf-8 -*-


import theano.tensor as T


class Optimizer(object):
    '''Abstract optimizer base class.'''


    def getUpdates(self, cost, layers):
        raise NotImplementedError()

    def getInputTensors(self):
        raise NotImplementedError()

    def getInputValues(self, nmEpochDone):
        raise NotImplementedError()

    def defaultGradParam(self, cost, defaultGradParams):
        grads = [T.grad(cost, param) for param in defaultGradParams]
        return grads