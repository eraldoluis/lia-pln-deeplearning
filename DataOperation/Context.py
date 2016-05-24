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
    def __init__(self,windowSize):
        '''
        :type windowSize: int
        :param windowSize: the size of window
        
        '''
        
        self.__windowSize = windowSize

    def buildWindows(self,objs,startPadding, endPadding = None ):
        '''
        :type objs: []
        :param objs: list of objects that will be used to build windows. Objects can be anything even lists.
        
        :param startPadding: Object that will be place when the initial limit of list is exceeded
        
        :param endPadding: Object that will be place when the end limit of objs is exceeded. 
            When this parameter is null, so the endPadding has the same value of startPadding

        Returns a window at a time
        '''
       
        if endPadding is None:
            endPadding = startPadding
        
        windows = []
        inputLen = len(objs)
        contextSize = int(math.floor((self.__windowSize - 1) / 2))
        
        for idx in range(inputLen):
            context =[] 
            i = idx - contextSize
            
            while (i <= idx + contextSize):
                if(i < 0):
                    context.append(startPadding)
                elif (i >= inputLen):
                    context.append(endPadding)
                else:
                    context.append(objs[i])
                i += 1

            yield  context
        
        
        
        
    