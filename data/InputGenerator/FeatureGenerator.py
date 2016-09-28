#!/usr/bin/env python
# -*- coding: utf-8 -*-


class FeatureGenerator(object):
    """
    Interface of a class that generates the features for the training.
    """

    def __call__(self, data):
        """
        :param data: one example returned by DatasetReader.read()
        :return: a list of features in textual format
        """
        return self.generate(data)

    def generate(self, data):
        """
        :param data: data from the data set. This data can be raw attributes or labels of one or more examples.
        :return: a list of inputs
        """
        not NotImplementedError()
