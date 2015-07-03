#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NNet.Util import WeightTanhGenerator
import codecs

class WordVector:
    UNKOWN_INDEX = 0
    
    def __init__(self,filePath='',wordSize=-1):        
        self.__wordVecs = []
        self.__generatorWeight = WeightTanhGenerator()
        if filePath:
            self.putUsingFile(filePath)
        else:
            self.__wordSize = wordSize
            
    def append(self,wordVector):            
        if  wordVector is None:
            wordVector = self.__generatorWeight.generateVector(self.__wordSize)
        else:
            if self.__wordSize < 1:
                self.__wordSize = len(wordVector)
            
            if self.__wordSize != len(wordVector):
                raise Exception("O vetor a ser adicionado tem um tamanho de" + str(len(wordVector)) +  " que " + 
                            "é diferente do tamanho dos outros vetores " + str(self.__wordSize) + " index " + str(self.getLength()))
        
        self.__wordVecs.append(wordVector)
    
    def putUsingFile(self,vecfilename):
        fVec = codecs.open(vecfilename, 'r','utf-8')
        for line in fVec:
            self.putWordVecStr(line)
        fVec.close()
        
    def putWordVecStr(self,str):
        return self.append([float(num) for num in str.split()])
    
    
    def getWordVector(self,index):
        return self.__wordVecs[index]
        
    def getLength(self):
        return len(self.__wordVecs)
    
    def getLenWordVector(self):
        return self.__wordSize
    
    def getWordVectors(self):
        return self.__wordVecs
