#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Window:
    """
    Generate a sequence of windows (sliding window features) from a sequence of items.
    """

    def __init__(self, embedding, windowSize, startPadding, endPadding=None):
        """
        :type windowSize: int
        :param windowSize: the size of window

        :type startPadding: string
        :param startPadding: string to extend the given sequences to the left.

        :type endPadding: string
        :param endPadding: string to extend the given sequences to the right.
            If it is not given, the start padding will be used.
        """
        if windowSize % 2 == 0:
            raise Exception("The window size (%s) is not odd." % windowSize)
        self.__windowSize = windowSize

        # Verify whether the start padding symbol exist in the lexicon.
        if not embedding.exist(startPadding):
            if embedding.isReadOnly():
                raise Exception("Start padding symbol does not exist!")
            startPaddingIdx = embedding.put(startPadding)
        else:
            startPaddingIdx = embedding.getLexiconIndex(startPadding)
        self.__startPaddingIdx = startPaddingIdx

        # Verify whether the end padding symbol exist in the lexicon.
        endPaddingIdx = startPaddingIdx
        if endPadding:
            if not embedding.exist(endPadding):
                if embedding.isReadOnly():
                    raise Exception("End padding symbol does not exist!")
                endPaddingIdx = embedding.put(endPadding)
            else:
                endPaddingIdx = embedding.getLexiconIndex(endPadding)
        self.__endPaddingIdx = endPaddingIdx

    def buildWindows(self, sequence):
        """
        Receives a sequence of items and returns a sequence of windows for the given sequence.

        :type sequence: list
        :param sequence: sequence of items for which the windows will be created.

        :return Returns a generator from yield
        """
        startPadding = self.__startPaddingIdx
        endPadding = self.__endPaddingIdx

        winSize = self.__windowSize
        contextSize = (winSize - 1) / 2

        # Extend the given sequence with start and end padding items.
        paddedSequence = [startPadding] * contextSize + sequence + [endPadding] * contextSize

        windows = []
        for idx in xrange(len(sequence)):
            windows.append(paddedSequence[idx:idx + winSize])

        return windows
