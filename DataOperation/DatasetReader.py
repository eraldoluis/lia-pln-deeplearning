#!/usr/bin/env python
# -*- coding: utf-8 -*-

class DatasetReader(object):
    """
    The interface of data set readers.
    """

    def read(self):
        '''
        :return: a generator(created by yield) that will return attributes and labels
            of one or more examples
        '''
        raise NotImplementedError()
