#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Contains a set of classes that read  attributes and/or label of tokens from a data set
'''

import codecs
import logging

from data.DatasetReader import DatasetReader


"""
Read sentences without labels. Each line comprises a sentence.
"""
class TokenReader(DatasetReader):
    def __init__(self, filePath, sep=None):
        """
        :type filePath: String
        :param filePath: dataset path
        
        :type sep: string
        :param sep: character or string which separates tokens
        """
        self.__filePath = filePath
        self.__sep = sep
        self.__log = logging.getLogger(__name__)
        self.__printedNumberTokensRead = False

    def read(self):
        """
        Returns a tuple  of tokens in a row
        """
        f = codecs.open(self.__filePath, "r", "utf-8")
        nmTokens = 0

        for line in f:
            tokens = line.strip().split(self.__sep)
            noneLabels = [None] * len(tokens)
            yield (tokens, noneLabels)

            nmTokens += len(tokens)

        if not self.__printedNumberTokensRead:
            self.__log.info("Number of tokens read: %d" % nmTokens)


"""
Read labeled sentences. Each token is associated with a label. Each sentence is
presented in one line, following a format like:

    The_ART man_SUBS is_VERB tall_ADJ

In this example, token separator is space and token/label separator is underline.
"""
class TokenLabelReader(DatasetReader):

    def __init__(self, filePath, labelTknSep, sep=None, oneTokenPerLine=False):
        """
        :type filePath: String
        :param filePath: dataset path

        :type labelTknSep: string
        :param labelTknSep: character or string which separate token from label

        :type sep: string
        :param sep: character or string which separates the expressions formed by token and label

        :type oneTokenPerLine: boolean
        :param oneTokenPerLine: if true then each line contains one token 
            (a blank line separates sentences). Otherwise (default), each line
            contains a full sentence (there is two separators: one for features
            and another for tokens). In the former case, only labelTknSep is 
            used. The argument sep is ignored.

        In example "The_ART man_SUBS is_VERB tall_ADJ", labelTknSep = "_" and  sep = " "
        """
        self.__filePath = filePath
        self.__labelTknSep = labelTknSep
        self.__sep = sep

        self.__log = logging.getLogger(__name__)
        self.__printedNumberTokensRead = False

    def read(self):
        """
        Returns a list of tokens and labels in a row at time
        """
        f = codecs.open(self.__filePath, "r", "utf-8")
        nmTokens = 0

        for line in f:
            tknLabelSets = line.strip().split(self.__sep)
            tkns = []
            labels = []

            for tknLabelSet in tknLabelSets:
                tkn, label = tknLabelSet.rsplit(self.__labelTknSep, 1)

                if not len(tkn):
                    raise Exception("It was found an empty token")

                if not len(label):
                    raise Exception("It was found an empty label")

                tkns.append(tkn)
                labels.append(label)

            nmTokens += len(tkns)

            assert len(tkns) == len(labels)
            yield (tkns, labels)

        if not self.__printedNumberTokensRead:
            self.__log.info("Number of tokens read: %d" % nmTokens)


"""
Read labeled sentences. Each token is associated with a label. The input format
must be like:

    The ART
    man SUBS
    is VERB
    tall ADJ

Where each line contains a token and its label separated by space.
"""
class TokenLabelPerLineReader(DatasetReader):
    def __init__(self, filePath, labelTknSep):
        """
        :type filePath: String
        :param filePath: dataset path

        :type labelTknSep: string
        :param labelTknSep: character or string which separate token from label
        """
        self.__filePath = filePath
        self.__labelTknSep = labelTknSep

        self.__log = logging.getLogger(__name__)
        self.__printedNumberTokensRead = False

    def read(self):
        """
        Returns a list of tokens and labels in a row at time
        """
        f = codecs.open(self.__filePath, "r", "utf-8")
        nmTokens = 0

        # Tokens and their labels for a whole sentence.
        tkns = []
        labels = []

        for line in f:
            line = line.strip()

            if len(line) == 0:
                # Blank line separates sentences.
                yield (tkns, labels)

            tknLabel = line.split(self.__sep)
            tkns.append(tknLabel[0])
            labels.append(tknLabel[1])

            nmTokens += 1

            assert len(tkns) == len(labels)

        if len(tkns) > 0:
            # Last sentence.
            yield (tkns, labels)

        if not self.__printedNumberTokensRead:
            self.__log.info("Number of tokens read: %d" % nmTokens)
