#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

'''
Contains classes that get the context from an object, like window of words
'''


class Window:
    '''
    Window of words, characters or any kind of objects.
    '''

    def __init__(self, windowSize):
        '''
        :type windowSize: int
        :param windowSize: the size of window
        
        '''

        self.__windowSize = windowSize

        if self.__windowSize % 2 == 0:
            raise Exception("The window size is odd.")

    def buildWindows(self, objs, startPadding, endPadding=None):
        '''
        Receives a list of objects and creates the windows of this list.

        :type objs: []
        :param objs: list of objects that will be used to build windows. Objects can be anything even lists.
        
        :param startPadding: Object that will be place when the initial limit of list is exceeded
        
        :param endPadding: Object that will be place when the end limit of objs is exceeded. 
            When this parameter is null, so the endPadding has the same value of startPadding

        :return Returns a generator from yield
        '''

        if endPadding is None:
            endPadding = startPadding

        windows = []
        inputLen = len(objs)
        contextSize = int(math.floor((self.__windowSize - 1) / 2))

        for idx in range(inputLen):
            context = []
            i = idx - contextSize

            while (i <= idx + contextSize):
                if (i < 0):
                    context.append(startPadding)
                elif (i >= inputLen):
                    context.append(endPadding)
                else:
                    context.append(objs[i])
                i += 1

            yield context


    @staticmethod
    def checkPadding(startPadding, endPadding, embedding):
        '''
        Verify if the start padding and end padding exist in lexicon or embedding.

        :param startPadding: Object that will be place when the initial limit of a list is exceeded

        :param endPadding: Object that will be place when the end limit a list is exceeded.
            If this parameter is None, so the endPadding has the same value of startPadding

        :param embedding: DataOperation.Embedding.Embedding

        :return: the index of start and end padding in lexicon
        '''

        if not embedding.exist(startPadding):
            if embedding.isStopped():
                raise Exception("Start Padding doens't exist")

            startPaddingIdx = embedding.put(startPadding)
        else:
            startPaddingIdx = embedding.getLexiconIndex(startPadding)
        if endPadding is not None:
            if not embedding.exist(endPadding):
                if embedding.isStopped():
                    raise Exception("End Padding doens't exist")

                endPaddingIdx = embedding.put(endPadding)
            else:
                endPaddingIdx = embedding.getLexiconIndex(endPadding)
        else:
            endPaddingIdx = None

        return startPaddingIdx, endPaddingIdx