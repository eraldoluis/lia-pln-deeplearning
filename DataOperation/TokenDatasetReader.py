#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Contains a set of classes that read  attributes and/or label of tokens from data set
'''

import codecs


class Reader(object):
    def read(self):
        '''
        Return attributes or/and labels of tokens in a row at time 
        '''
        raise NotImplementedError()


class TokenReader(Reader):
    """
    Read unlabel dataset tokens 
    """

    def __init__(self, filePath, sep=' '):
        """
        :type filePath: String
        :param filePath: dataset path
        
        :type sep: string
        :param sep: character or string which separate tokens
        """
        self.__filePath = filePath
        self.__sep = sep

    def read(self):
        """
        Return a tuple  of tokens in a row at time
        """
        f = codecs.open(self.__filePath, "r", "utf-8")

        for line in f:
            yield (line.strip().split(self.__sep),None)


class TokenLabelReader(Reader):
    """
    Read unlabel dataset tokens
    """

    def __init__(self, filePath, labelTknSep, sep=' '):
        """
        :type filePath: String
        :param filePath: dataset path

        :type labelTknSep: string
        :param labelTknSep: character or string which separate token from label

        :type sep: string
        :param sep: character or string which separate token and label sets
        """
        self.__filePath = filePath
        self.__labelTknSep = labelTknSep
        self.__sep = sep

    def read(self):
        """
        Return a list of tokens and labels in a row at time
        """
        f = codecs.open(self.__filePath, "r", "utf-8")

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

            assert len(tkns) == len(labels)
            yield (tkns,labels)