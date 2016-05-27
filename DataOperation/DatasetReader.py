#!/usr/bin/env python
# -*- coding: utf-8 -*-

class DatasetReader(object):
    """
    The interface of the classes that will read the dataset.
    """

    def read(self):
        '''
        :return: a generator(created by yield) that will return features and labels of the data set.
        This features and labels are the same data found in the data set, in other words, these datas are not processed.
        '''
        raise NotImplementedError()
