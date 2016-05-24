#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy

from keras import initializations


def generateRandomNumberUniformly(low, high, n_in, n_out):
    if n_out == 0.0:
        return numpy.random.uniform(low, high, n_in)
    else:
        return numpy.random.uniform(low, high, (n_in, n_out))


class WeightGenerator(object):

    def generateWeight(self,shape):
        '''
        :type shape: tuple
        :param shape: a tuple like numpy

        :return: matrix with the weights
        '''
        raise NotImplementedError()


class GlorotUniform(WeightGenerator):
    """
    This initialization can be use with tanh.
    """

    def generateWeight(self, shape):
        """
        :param shape:
        :return:
        """
        high = numpy.sqrt(6. / (shape[0] + shape[1]))
        return generateRandomNumberUniformly(-high, high, shape[0],shape[1])


class SigmoidGenerator(WeightGenerator):
    def generateWeight(self, shape):
        #   results presented in [Xavier10] suggest that you
        #   should use 4 times larger initial weights for sigmoid compared to tanh
        #

        high = numpy.sqrt(6. / (shape[0] + shape[1]))
        return 4 * generateRandomNumberUniformly(-high, high, shape[0], shape[1])

