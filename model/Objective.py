#!/usr/bin/env python
# -*- coding: utf-8 -*-


import theano
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


class WeightedNegativeLogLikelihood(Objective):
    def __init__(self, weights=None):
        """
        :param weights: array of weights for each class.
        """
        if weights is None:
            self.__weights = None
        else:
            self.__weights = T.as_tensor_variable(weights)

    def calculateError(self, output, ypred, ytrue):
        weights = T.reshape(self.__weights, (1,-1))
        weights = T.repeat(weights, output.shape[0], axis = 0)
        
        return -T.mean(T.log(
            output[T.arange(ytrue.shape[0]), ytrue] * weights[T.arange(ytrue.shape[0]), ytrue]
        ))

class NegativeLogLikelihoodOneExample(Objective):
    def __init__(self, weights=None):
        """
        :param weights: array of weights for each class.
        """
        if weights is None:
            self.__weights = None
        else:
            self.__weights = T.as_tensor_variable(weights)

    def calculateError(self, output, ypred, ytrue):
        if self.__weights is not None:
            return -T.log(output[ytrue]) * self.__weights[ytrue]
        return -T.log(output[ytrue])
