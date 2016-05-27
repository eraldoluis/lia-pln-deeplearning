#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Queue
import random
import threading

from Context import context
import numpy
from DataOperation.TokenDatasetReader import TokenReader, TokenLabelReader
from datetime import time

'''
Contains classes that format the data to can be read by the supervised algorithms
'''








class HiddenLayerInpWnnGenerator(object):
    '''
    This class was created specially to control the use of memory.
    "generator"  function will return a generator which will return training input to hidden layer.
    You can pass this generator to the fit_generator of keras.
    '''

    def __init__(self, allWindows, embedding, batchSize, windowSize):
        self.lock = threading.Lock()
        self.batchSize = batchSize
        self.allWindows = allWindows
        self.batchIdxs = self.__generateBatches(allWindows, batchSize)
        self.sortedBatchIdxs = []
        self.allWindows = allWindows
        self.embedding = embedding
        self.batchSize = batchSize
        self.windowSize = windowSize

    def __generateBatches(self, allWindows, batchSize):
        batchIdx = 0
        numExs = len(allWindows)

        if batchSize > 0:
            numBatches = numExs / batchSize
            if numBatches <= 0:
                numBatches = 1
                batchSize = numExs
            elif numExs % batchSize > 0:
                numBatches += 1
        else:
            numBatches = 1
            batchSize = numExs

        return range(numBatches)

    def generator(self):
        '''
        Create a generator that will return training input to hidden layer.
        '''
        while 1:
            with self.lock:
                if len(self.sortedBatchIdxs) == 0:
                    random.shuffle(self.batchIdxs)

                    for batchIdx in self.batchIdxs:
                        self.sortedBatchIdxs.append(batchIdx)

                batchIdx = self.sortedBatchIdxs.pop(0)

            windows = self.allWindows[batchIdx * self.batchSize: (batchIdx + 1) * self.batchSize]
            batchInputs = numpy.zeros((len(windows), self.windowSize * self.embedding.getEmbeddingSize()))

            for i, window in enumerate(windows):
                example = batchInputs[i]
                j = 0

                for tknIdx in window:
                    for elem in self.embedding.getEmbeddingByIndex(tknIdx):
                        example[j] = elem
                        j += 1

            yield (batchInputs, batchInputs)


