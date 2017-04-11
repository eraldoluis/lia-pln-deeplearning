#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random


class BatchIteratorUnion(object):
    """
    This class is responsible for creating a artificial dataset from the combination of many datasets.
    To get a example, you can use the method getRandomly which sort randomly a example from the datasets or
        you can use as a iterator.
    """

    def __init__(self, listSyncBatchList, shuffle=True):
        """
        :param listSyncBatchList: list that contains SyncBatchIterator from each dataset
        :param shuffle: If this parameter is true so it shuffles the examples.
        """
        self.__list = listSyncBatchList
        self.__shuffle = shuffle

        # Store sets compound by example id and the batch iterator id where this example came from. [(batch_idx, example_idx)]
        self.__batches = []
        # Store id of each set in "batches".
        self.__batchIdxs = []
        # Point to the next element to be removed from batchIdxs.
        self.__current = 0

        for batchIteratorIdx, batchIterator in enumerate(self.__list):
            for exampleIdx in range(batchIterator.size()):
                self.__batchIdxs.append(len(self.__batches))
                self.__batches.append((batchIteratorIdx, exampleIdx))

        if self.__shuffle:
            random.shuffle(self.__batchIdxs)

    def getSize(self):
        return len(self.__batches)

    def __iter__(self):
        return self

    def next(self):
        """
        :return: (batch iterator id, example id)
        """
        if self.__current < len(self.__batchIdxs):
            batchIteratorIdx, exampleIdx = self.__batches[self.__batchIdxs[self.__current]]
            self.__current += 1

            return batchIteratorIdx, self.__list[batchIteratorIdx].get(exampleIdx)
        else:
            if self.__shuffle:
                random.shuffle(self.__batchIdxs)

            self.__current = 0

            raise StopIteration()

    def getRandomly(self):
        """
        This method returns the index of dataset which the example was taken and the example.
        The example are randomly sorted.
        WARNING: This method can return a example which were already sorted.
        :return: the index of dataset which the example was removed and the example.
        """
        i = random.randint(0, self.getSize())
        batchIteratorIdx, exampleIdx = self.__batches[i]

        return batchIteratorIdx, self.__list[batchIteratorIdx].get(exampleIdx)
