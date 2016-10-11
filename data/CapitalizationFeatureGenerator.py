#!/usr/bin/env python
# -*- coding: utf-8 -*-


from data.FeatureGenerator import FeatureGenerator
from data.Window import Window


class CapitalizationFeatureGenerator(FeatureGenerator):
    """
    Return the capitalization feature of each word in a window.
    The capitalization feature has five possible values: all lowercased, first uppercased, all uppercased,
    contains an uppercased letter, and all other cases. Each possible value is represented by a specific embedding which
    is inside a lookup table.
    """

    def __init__(self, windowSize, lexicon, filters=None):
        """
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
        self.__lexicon = lexicon
        self.__filters = filters

        # Add all possible options
        for i in ["all_lower", "first_uppercased", "all_uppercased", "contains_uppercased", "N/A"]:
            lexicon.put(i)

    def generate(self, sequence):
        """
        Receives a sequence of tokens and returns a sequence of token windows.

        :type sequence: list[basestring]
        :param sequence: sequence of tokens
        :return: a sequence of token windows.
        """
        capIdxs = []

        for token in sequence:

            for f in self.__filters:
                token = f.filter(token, sequence)

            if token.islower():
                capFeature = "all_lower"
            elif token[0].isupper():
                capFeature = "first_uppercased"
            elif token.isupper():
                capFeature = "all_uppercased"
            elif self.__containUppercasedLetter(token):
                capFeature = "contains_uppercased"
            else:
                capFeature = "N/A"

            if capFeature is self.__lexicon.getUnknownIndex():
                raise Exception("A capitalization feature doesn't exist in lexicon.")

            capIdxs.append(self.__lexicon.put(capFeature))

        return self.__window.buildWindows(capIdxs)

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
