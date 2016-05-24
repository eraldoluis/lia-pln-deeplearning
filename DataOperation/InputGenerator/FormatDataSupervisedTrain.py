#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Queue
import random
import threading

from data_operation import context
import numpy
from data_operation.token_dataset_reader import TokenReader, TokenLabelReader
from datetime import time

'''
Contains classes that format the data to can be read by the supervised algorithms
'''


class AsyncReader(object):
    def __init__(self, generator, maxqSize=10, waitTime=0.05):
        self.__queue, self.__stop = self.generatorQueue(generator, maxqSize, waitTime)

    def __iter__(self):
        return self

    def next(self):
        b = self.__queue.get()

        if b is None:
            raise StopIteration()

        return b

    def generatorQueue(generator, max_q_size=10, wait_time=0.05):
        '''Builds a threading queue out of a data generator.
        Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
        '''
        q = Queue()
        _stop = threading.Event()

        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        try:
                            generator_output = generator.next()
                        except StopIteration:
                            generator_output = None
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception as e:
                    _stop.set()
                    print e
                    raise

        thread = threading.Thread(target=data_generator_task)
        thread.daemon = True
        thread.start()

        return q, _stop

class WNNInputBuilder:
    def __init__(self, windowSize, startPadding, endPadding=None):
        '''
        :type windowSize: int
        :param windowSize: the size of window

        :param startPadding: Object that will be place when the initial limit of list is exceeded

        :param endPadding: Object that will be place when the end limit of objs is exceeded.
            When this parameter is null, so the endPadding has the same value of startPadding
        '''
        self.__window = context.Window(windowSize)
        self.__startPadding = startPadding
        self.__endPadding = endPadding

    def __readFile(self, reader, embedding, labelLexicon, filters, separateBySentence):
        '''
        :param reader: TokenReader or TokenLabelReader

        :type embedding: embedding.Embedding
        :param embedding:

        :type lexicon: lexicon.Lexicon
        :param labelLexicon: a empty lexicon of the labels which will be filled

        :type filters: []
        :param filters: a list of filters. These filters modify the tokens, for instance the filter which transforms all case of words to lower case.

        :param separateBySentence: if this parameter is True, each element of x and y are word windows and labels of sentence.
                Else, each element of x and y are the word window and label of token.

        :return: a tuple (x,y), where x is a list of word windows and y are the labels.
        '''
        endPaddingIdx, startPaddingIdx = self.checkPadding(embedding)
        x = []
        y = []

        for tokens, labels in reader.read():
            tknIdxs = []

            for token in tokens:
                for f in filters:
                    token = f.filter(token)

                tknIdxs.append(embedding.put(token))

            if separateBySentence:
                x.append([window for window in self.__window.buildWindows(tknIdxs, startPaddingIdx, endPaddingIdx)])

                if labels is not None:
                    y.append([labelLexicon.put(label) for label in labels])
            else:
                for window in self.__window.buildWindows(tknIdxs, startPaddingIdx, endPaddingIdx):
                    x.append(window)

                if labels is not None:
                    for label in labels:
                        y.append(labelLexicon.put(label))

        return x, y

    def readTokenFile(self, filePath, embedding, filters):
        '''
        Reads a file which only contains tokens.

        :param filePath: path of unlabeled data set


        :param embedding:

        :type filters: []
        :param filters: a list of filters. These filters modify the tokens, for instance the filter which transforms all case of words to lower case.

        :returns list of inputs. These elements are word windows
        '''

        reader = TokenReader(filePath)

        return self.__readFile(reader, embedding, filters, False)

    def readTokenLabelFile(self, filePath, embedding, labelLexicon, filters, labelTknSep, separateBySentence):
        '''
        Reads a file which contains tokens and your labels.

        :param filePath: path of unlabeled data set

        :type embedding: embedding.Embedding
        :param embedding:

        :param labelLexicon: a empty lexicon of the labels which will be filled,

        :type filters: []
        :param filters: a list of filters. These filters modify the tokens, for instance the filter which transforms all case of words to lower case.

        :type labelTknSep: string
        :param labelTknSep: character or string which separate token from label

        :param separateBySentence: if this parameter is True, each element of x and y are word windows and labels of sentence.
                Else, each element of x and y are the word window and label of token.

        :returns list of inputs. These elements are word windows
        '''
        reader = TokenLabelReader(filePath, labelTknSep)

        return self.__readFile(reader, embedding, labelLexicon, filters, separateBySentence)

    def checkPadding(self, embedding):
        '''
        Verify if the start padding and end padding exist in lexicon or embedding.
        :param embedding: embedding.Embedding

        :return: the index of start and end padding in lexicon
        '''

        if not embedding.exist(self.__startPadding):
            if embedding.isStopped():
                raise Exception("Start Padding doens't exist")

            startPaddingIdx = embedding.put(self.__startPadding)
        else:
            startPaddingIdx = embedding.getLexiconIndex(self.__startPadding)
        if self.__endPadding is not None:
            if not embedding.exist(self.__endPadding):
                if embedding.isStopped():
                    raise Exception("End Padding doens't exist")

                endPaddingIdx = embedding.put(self.__endPadding)
            else:
                endPaddingIdx = embedding.getLexiconIndex(self.__endPadding)
        else:
            endPaddingIdx = None
        return endPaddingIdx, startPaddingIdx




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


