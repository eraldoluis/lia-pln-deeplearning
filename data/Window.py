#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math


class Window:
    """
    Window of words, characters or any kind of objects.
    """

    def __init__(self, windowSize):
        """
        :type windowSize: int
        :param windowSize: the size of window
        """
        if windowSize % 2 == 0:
            raise Exception("The window size (%s) is not odd." % windowSize)
        self.__windowSize = windowSize

    def buildWindows(self, sequence, startPadding, endPadding=None):
        """
        Receives a list of objects and creates the windows of this list.

        :type sequence: []
        :param sequence: sequence of items for which the windows will be created.
        
        :param startPadding: Object that will be placed when the initial limit of list is exceeded
        
        :param endPadding: Object that will be placed when the end limit of sequence is exceeded.
            When this parameter is null, so the endPadding has the same value of startPadding

        :return Returns a generator from yield
        """
        if endPadding is None:
            endPadding = startPadding

        winSize = self.__windowSize
        contextSize = (winSize - 1) / 2

        # Extend the given sentence with start and end padding items.
        paddedSequence = [startPadding] * contextSize + sequence + [endPadding] * contextSize

        windows = []
        for idx in xrange(len(sequence)):
            windows.append(paddedSequence[idx:idx + winSize])

        return windows

    @staticmethod
    def checkPadding(startPadding, endPadding, embedding):
        """
        Verify if the start padding and end padding exist in lexicon or embedding.

        :param startPadding: Object that will be place when the initial limit of a list is exceeded

        :param endPadding: Object that will be place when the end limit a list is exceeded.
            If this parameter is None, so the endPadding has the same value of startPadding

        :param embedding: DataOperation.Embedding.Embedding

        :return: the index of start and end padding in lexicon
        """

        if not embedding.exist(startPadding):
            if embedding.isReadOnly():
                raise Exception("Start Padding doens't exist")

            startPaddingIdx = embedding.put(startPadding)
        else:
            startPaddingIdx = embedding.getLexiconIndex(startPadding)

        endPaddingIdx = None
        if endPadding is not None:
            if not embedding.exist(endPadding):
                if embedding.isReadOnly():
                    raise Exception("End Padding doens't exist")

                endPaddingIdx = embedding.put(endPadding)
            else:
                endPaddingIdx = embedding.getLexiconIndex(endPadding)

        return startPaddingIdx, endPaddingIdx
