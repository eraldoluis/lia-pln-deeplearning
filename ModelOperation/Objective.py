#!/usr/bin/env python
# -*- coding: utf-8 -*-


import theano.tensor as T
from theano import printing


class Objective(object):
    def calculateError(self, output, ypred, ytrue):
        raise NotImplementedError()


class MeanSquaredError(Objective):
    def calculateError(self, output, ypred, ytrue):
        return T.mean(T.square(ypred - ytrue))


class NegativeLogLikelihood(Objective):
    def calculateError(self, output, ypred, ytrue):
        return -T.mean(T.log(output)[T.arange(ytrue.shape[0]), ytrue])


