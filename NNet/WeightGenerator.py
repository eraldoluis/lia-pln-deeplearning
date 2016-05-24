#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras import initializations

class Initialization(object):

    def initialize(self,shape):
        '''
        :type shape: tuple
        :param shape: a tuple like numpy

        :return: matrix with the weights
        '''
        raise NotImplementedError()


class TanhInilization(Initialization):

    def generateWeight(self, n_in, n_out):
        high = numpy.sqrt(6. / (n_in + n_out))
        return generateRandomNumberUniformly(-high, high, n_in, n_out)

    def initialize(self, shape):
        if len(shape) > 2:



def generateRandomNumberUniformly(low, high, n_in, n_out):
    if n_out == 0.0:
        return numpy.random.uniform(low, high, (n_in))
    else:
        return numpy.random.uniform(low, high, (n_in, n_out))

class WeightTanhGenerator:
