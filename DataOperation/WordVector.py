#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''

from NNet.Util import WeightTanhGenerator

class WordVector:
    UNKOWN_INDEX = 0
    
    def __init__(self,filePath='',wordSize=-1):        
        self.__wordVecs = [[]]
        self.__generatorWeight = WeightTanhGenerator()
        if filePath:
            self.putUsingFile(filePath)
        else:
            self.__wordSize = wordSize
            
            if self.__wordSize > 0:
                self.__wordVecs[0] = self.__generatorWeight.generateVector(self.__wordSize)
            
    def append(self,wordVector):
        
        if self.__wordSize < 0:
            self.__wordSize = len(wordVector)
            self.__wordVecs[0] = self.__generatorWeight.generateVector(self.__wordSize)
        
        if self.__wordSize != len(wordVector):
            raise Exception("O vetor a ser adicionado tem um tamanho de" + str(len(wordVector)) +  " que " + 
                            "é diferente do tamanho dos outros vetores " + str(self.__wordSize) + " index " + str(self.getLength()))
        
        if  wordVector is None:
            wordVector = self.__generatorWeight.generateVector(self.__wordSize)
            
        self.__wordVecs.append(wordVector)
    
    def putUsingFile(self,vecfilename):
        fVec = open(vecfilename, 'r')
        for line in fVec:
            self.putWordVecStr(line)
        fVec.close()
        
    def putWordVecStr(self,str):
        return self.append([float(num) for num in str.split()])
    
        
    def getLength(self):
        return len(self.__wordVecs)
    
    def getLenWordVector(self):
        return self.__wordSize
    
    def getWordVectors(self):
        return self.__wordVecs
