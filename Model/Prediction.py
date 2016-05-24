#!/usr/bin/env python
# -*- coding: utf-8 -*-


import theano.tensor as T

class Prediction(object):

    def predict(self,output):
        not NotImplementedError()


class ArgmaxPrediction(Prediction):

    def __init__(self,axis):
        self.__axis = axis

    def predict(self,output):
        return T.argmax(output,self.__axis)