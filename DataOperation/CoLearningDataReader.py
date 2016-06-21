#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from DataOperation.DatasetReader import DatasetReader
from DataOperation.TokenDatasetReader import TokenLabelReader, TokenReader


class CoLearningDataReader(DatasetReader):

    def __init__(self, labelDataset, unlabelDataset, labelTknSep, sep=None):
        """
        :type filePath: String
        :param filePath: dataset path

        :type sep: string
        :param sep: character or string which separates tokens
        """
        self.__tokenLabelReader = TokenLabelReader(labelDataset,labelTknSep,sep)
        self.__tokenReader = TokenReader(unlabelDataset, sep)

        self.__log = logging.getLogger(__name__)

    def read(self):
        generatoSup = self.__tokenLabelReader.read()

        for i in generatoSup:
            yield i

        generatoUnsup = self.__tokenReader.read()

        for i in generatoUnsup:
            yield i
