#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''

from NNet.Util import WeightTanhGenerator

class WordVector:
    UNKOWN_INDEX = 0
    
    def __init__(self,filePath='',wordSize=-1):        
        self.__wordVecs = []
        self.__generatorWeight = WeightTanhGenerator()
        
        
        if filePath:
            self.append([])
            self.putUsingFile(filePath)
            self.setLenWordVectorAuto()
        else:
            self.__wordSize = wordSize
            
            if self.__wordSize > 0:
                self.append(self.__generatorWeight.generateVector(self.__wordSize))
            else:
                self.append([])

    def append(self,wordVector):
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
    
    def setLenWordVectorAuto(self):
        self.__wordSize = len(self.__wordVecs[1])
        self.__wordVecs[0] = self.__generatorWeight.generateVector(self.__wordSize)
        
    def getLength(self):
        return len(self.__wordVecs)
    
    def getLenWordVector(self):
        return self.__wordSize
    
    def getWordVectors(self):
        return self.__wordVecs
