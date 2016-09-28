#!/usr/bin/env python
# -*- coding: utf-8 -*-


import theano.tensor as T

class Objective(object):
    def calculateError(self, output, ypred, ytrue):
        raise NotImplementedError()


class MeanSquaredError(Objective):
    def calculateError(self, output, ypred, ytrue):
        return T.mean(T.sum(T.square(ypred - ytrue),axis=1))


class NegativeLogLikelihood(Objective):
    def calculateError(self, output, ypred, ytrue):
        return -T.mean(T.log(output[T.arange(ytrue.shape[0]), ytrue]))

class NegativeLogLikelihoodOneExample(Objective):
    def calculateError(self, output, ypred, ytrue):
        return -T.log(output[ytrue])
