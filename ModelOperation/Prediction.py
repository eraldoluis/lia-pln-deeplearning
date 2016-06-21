#!/usr/bin/env python
# -*- coding: utf-8 -*-


import theano.tensor as T
from theano import printing


class Prediction(object):
    def predict(self, output):
        not NotImplementedError()


class CoLearningWnnPrediction(Prediction):
    def predict(self, output):
        output = T.stack(output)
        return T.argmax(output, 2)[T.argmax(T.max(output, 2), 0),T.arange(output.shape[1])]


class ArgmaxPrediction(Prediction):
    def __init__(self, axis):
        self.__axis = axis

    def predict(self, output):
        return T.argmax(output, self.__axis)
