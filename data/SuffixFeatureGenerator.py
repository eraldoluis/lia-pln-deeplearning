#!/usr/bin/env python
# -*- coding: utf-8 -*-


from data.FeatureGenerator import FeatureGenerator
from data.Window import Window


class SuffixFeatureGenerator(FeatureGenerator):
    """
    Return the suffix feature of each word in a window.
    Each possible value of the suffix is represented by a specific embedding which
    is inside a lookup table.
    """

    def __init__(self, suffixSize, windowSize, lexicon, filters=None):
        """

        :param suffixSize: the size of suffix.

        :type windowSize: int
        :param windowSize: the size of window

        :type lexicon: data.Lexicon.Lexicon
        :param lexicon:

        :type filters: list[DataOperation.Filters.Filter]
        :param filters:
        """

        # All padding will have the feature PADDING
        startPadding = "PADDING"

        self.__window = Window(lexicon, windowSize, startPadding)
        self.__suffixSize = suffixSize
        self.__lexicon = lexicon
        self.__filters = filters

    def generate(self, sequence):
        """
        Receives a sequence of tokens and returns a sequence of token windows.

        :type sequence: list[basestring
        :param sequence: sequence of tokens
        :return: a sequence of token windows.
        """
        suffixIdxs = []

        for token in sequence:

            for f in self.__filters:
                token = f.filter(token, sequence)

            if len(token) <= self.__suffixSize:
                suffixFeatureIdx = self.__lexicon.getUnknownIndex()
            else:
                suffixFeatureIdx = self.__lexicon.put(token[-self.__suffixSize:])

            suffixIdxs.append(suffixFeatureIdx)

        return self.__window.buildWindows(suffixIdxs)

    @staticmethod
    def __containUppercasedLetter(token):
        """
        Returns true if the word contains an uppercased letter.

        :type token: basestring
        :param token:
        :return: boolean
        """
        for c in token:
            if c.isupper():
                return True

        return False
