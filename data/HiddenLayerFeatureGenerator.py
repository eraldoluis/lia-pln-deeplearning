#!/usr/bin/env python
# -*- coding: utf-8 -*-
from data.FeatureGenerator import FeatureGenerator

from data.WordWindowGenerator import WordWindowGenerator

"""
Contains classes that format the data to can be read by the supervised algorithms
"""


class HiddenLayerFeatureGenerator(FeatureGenerator):
    def __init__(self, windowSize, embedding, filters, startPadding, endPadding):
        self.embedding = embedding
        self.windowSize = windowSize
        self.__windowGenerator = WordWindowGenerator(windowSize, embedding, filters,
                                                     startPadding, endPadding)

    def generate(self, rawData):
        windowData = self.__windowGenerator.generate(rawData)
        hiddenData = []

        for window in windowData:
            i = []

            for lexiconIdx in window:
                for x in self.embedding.getEmbeddingByIndex(lexiconIdx):
                    i.append(x)

            hiddenData.append(i)

        return hiddenData