#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

from data.WordWindowGenerator import WordWindowGenerator
import numpy as np


class WordWindowGeneratorWithSubsampling(WordWindowGenerator):
    """
    Generate window of words from each word of a list.
    In this generator, we use subsampling, as [Mikolov 2013], which discard
    a window with the following probability: p= 1 - square(t/p(w)), where t is constant
    and p(w) is probability of the word(Mikolov call this probability as frequency).

    The equation in the paper is different of the equation used in word2vec. See more in:
    https://groups.google.com/forum/#!topic/word2vec-toolkit/pE2qLbgpuys
    We prefer to use the equation from the paper.

    """

    def __init__(self, t, probabilities, windowSize, lexicon, filters, startPadding, endPadding=None):
        super(WordWindowGeneratorWithSubsampling, self).__init__(windowSize, lexicon, filters,
                                                                 startPadding, endPadding)

        self.__probabilities = probabilities
        self.__t = t

        # I needed to create this struct because
        # I didn't find other way to communicate NoiseWordWindowGenerator
        # which examples wasn't discarded or not by WordWindowGeneratorWithSubsampling.
        #
        self.__windowsNotDiscardedByExamples = {}

    def doDiscard(self, tokenId):
        probability = self.__probabilities(tokenId)
        discardProbability = 1 - np.sqrt(self.__t / probability)
        discard = np.random.uniform() < discardProbability

        return discard

    def generate(self, sequence):
        windows = super(WordWindowGeneratorWithSubsampling, self).generate(sequence)

        windowsToReturn = []

        for window in windows:
            centerWord = math.floor(len(window) / 2)

            if self.doDiscard(window[centerWord]):
                continue

            windowsToReturn.append(window)

        return windowsToReturn
