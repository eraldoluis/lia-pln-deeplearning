#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


class Sampler(object):
    """
    It draws samples following a distribution probability.
    """

    def __init__(self, probabilities):
        self.__prob = np.asarray(probabilities)

        if round(self.__prob.sum(), 2) != 1.0:
            raise Exception("The sum of all probabilities isn't 1")

        self.__capacity = 2 ^ 20
        self.__samples = np.random.choice(len(self.__prob), size=self.__capacity, p=self.__prob)

        # It pointers to the first not used sample
        self.__ptr = 0

    def sample(self, nm=1):
        """
        :param nm: numbers of samples
        :return: a numpy vector with the samples
        """
        if len(self.__samples) - self.__ptr < nm:
            newSamples = np.random.choice(len(self.__prob), size=self.__capacity, p=self.__prob)
            newPtr = nm - (len(self.__samples) - self.__ptr)

            samples = self.__samples[self.__ptr:] + newSamples[:newPtr]

            self.__samples = newSamples
            self.__ptr = newPtr
        else:
            oldPtr = self.__ptr
            self.__ptr += nm
            samples = self.__samples[oldPtr: self.__ptr]

        return samples