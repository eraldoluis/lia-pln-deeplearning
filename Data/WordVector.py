#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''

from NNet.Util import WeightTanhGenerator

class WordVector:
    
    def __init__(self,wordSize=-1):        
        self.__wordVecs = []
        self.__generatorWeight = WeightTanhGenerator()
        self.__wordSize = wordSize

    def append(self,wordVector):
        if  wordVector is None:
            wordVector = self.__generatorWeight.generateVector(self.__wordSize)
            
        self.__wordVecs.append(wordVector)
        
    def setLenWordVectorAuto(self):
        self.__wordSize = len(self.__wordVecs[0])
        
    def setLenWordVector(self):
        self.__wordSize = len(self.__wordVecs[0])
        
    def getLength(self):
        return len(self.__wordVecs)
    
    def getLenWordVector(self):
        return self.__wordSize
    
    def getWordVectors(self):
        return self.__wordVecs
      
    def startRandomAllVecs(self, dictSize):
        self.__wordVecs = self.__generatorWeight.generateWeight(dictSize,self.__wordSize)
