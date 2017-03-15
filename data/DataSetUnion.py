#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import random



class DataSetUnion(object):
    """
    This class is responsible for creating a artificial dataset from the combination of many datasets.
    To get a example, you need to use the method getRandomly, which sort randomly a example from the datasets.
    This method can return already which were sorted.
    """

    def __init__(self, listSyncBatchList):
        """
        :type inputGenerators: list[DataOperation.InputGenerator.BatchIterator.SyncBatchIterator]
        :param listSyncBatchList: list that contains SyncBatchIterator from each dataset
        """
        self.__list = listSyncBatchList

        total = 0
        self.__ranges = []

        for d in self.__list:
            begin = total

            self.__ranges.append((begin, begin + d.size() - 1))

            total += d.size()

        self.__total = total

    def getSize(self):
        return self.__total

    def getRandomly(self):
        """
        This method returns the index of dataset which the example was taken and the example.
        The example are randomly sorted.
        WARNING: This method can return a example which were already sorted.
        :return: the index of dataset which the example was taken and the example.
        """
        i = random.randint(0, self.getSize())

        for idx, range in enumerate(self.__ranges):
            if range[0] <= i <= range[1]:
                return idx, self.__list[idx].get(i - range[0])