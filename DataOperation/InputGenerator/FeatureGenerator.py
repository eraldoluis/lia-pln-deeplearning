#!/usr/bin/env python
# -*- coding: utf-8 -*-


class FeatureGenerator(object):
    '''
    Interface of a class that generates the features for the training.
    '''

    def generate(self, rawData):
        '''
        :param rawData: data from the data set. This data can be raw attributes or labels of one or more examples.
        :return: a list of inputs
        '''
        not NotImplementedError()


