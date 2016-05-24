#!/usr/bin/env python
# -*- coding: utf-8 -*-


class FeatureGenerator(object):
    '''
    Interface of a class that generate the inputs for a training, which can be attributes or labels.
    '''

    def generate(self, rawData):
        '''
        :param rawData: data from the data set. This data can be raw attributes or labels of one or more examples.
        :return: a list of inputs
        '''
        not NotImplementedError()


