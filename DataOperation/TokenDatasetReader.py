#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Contains a set of classes that read  attributes and/or label of tokens from a data set
'''

import codecs
import logging

from DataOperation.DatasetReader import DatasetReader


class TokenReader(DatasetReader):
    """
    Read the tokens from a labeled dataset.
    """

    def __init__(self, filePath, sep=' '):
        """
        :type filePath: String
        :param filePath: dataset path
        
        :type sep: string
        :param sep: character or string which separates tokens
        """
        self.__filePath = filePath
        self.__sep = sep

    def read(self):
        """
        Returns a tuple  of tokens in a row
        """
        f = codecs.open(self.__filePath, "r", "utf-8")

        for line in f:
            tokens = line.strip().split(self.__sep)
            yield (tokens, tokens)


class TokenLabelReader(DatasetReader):
    """
    Reads the tokens and label from a labeled data set.
    """

    def __init__(self, filePath, labelTknSep, sep=' '):
        """
        :type filePath: String
        :param filePath: dataset path

        :type labelTknSep: string
        :param labelTknSep: character or string which separate token from label

        :type sep: string
        :param sep: character or string which separates the expressions formed by token and label

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
                tkn, label = tknLabelSet.split(self.__labelTknSep)

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
