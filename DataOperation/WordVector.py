#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NNet.Util import FeatureVectorsGenerator
import codecs
import numpy as np

# TODO: comentar esta classe
class WordVector:
    UNKOWN_INDEX = 0
    
    def __init__(self,filePath='',wordSize=-1,mode=None):        
        self.__wordVecs = []
        self.__generatorWeight = FeatureVectorsGenerator()
        if filePath:
            self.putUsingFile(filePath)
        else:
            self.__wordSize = wordSize
        
        self.mode = mode
        self.__len = 0
                
            
    def append(self,wordVector):            
        if  wordVector is None:
            if self.mode =='zeros':
                wordVector = np.zeros(self.__wordSize)
            elif self.mode == 'randomAll':
                self.__len +=1
                return
            else:    
                wordVector = self.__generatorWeight.generateVector(self.__wordSize)
                
        else:
            if self.__wordSize < 1:
                self.__wordSize = len(wordVector)
            
            if self.__wordSize != len(wordVector):
                raise Exception("O vetor a ser adicionado tem um tamanho de" + str(len(wordVector)) +  " que " + 
                            "é diferente do tamanho dos outros vetores " + str(self.__wordSize) + " index " + str(self.getLength()))
                
            if self.mode == 'randomAll':
                self.__len +=1
                self.__wordVecs = np.append(self.__wordVecs,[wordVector], axis=0)
                
                return
                
        #print len(self.__wordVecs), len(wordVector)
        self.__wordVecs.append(wordVector)
        #self.__wordVecs = np.append(self.__wordVecs,[wordVector], axis=0)
        
    
    def putUsingFile(self,vecfilename):
        fVec = codecs.open(vecfilename, 'r','utf-8')
        for line in fVec:
            self.putWordVecStr(line)
        fVec.close()
        
    def putWordVecStr(self,string):
        self.append([float(num) for num in string.split()])
    
    
    def getWordVector(self,index):
        return self.__wordVecs[index]
        
    def getLength(self):
        return len(self.__wordVecs)
    
    def getLenWordVector(self):
        return self.__wordSize
    
    def getWordVectors(self):
        return self.__wordVecs
    
    def startAllRandom(self):
        self.__wordVecs = self.__generatorWeight.generateWeight(self.__len,self.__wordSize)
        
    def minMax(self,norm_coef):
        self.__wordVecs = norm_coef *(self.__wordVecs - np.min(np.asarray(self.__wordVecs),axis=0))/np.ptp(np.asarray(self.__wordVecs),axis=0)
        
    def zScore(self,norm_coef):
        self.__wordVecs = norm_coef * (self.__wordVecs - np.mean(np.asarray(self.__wordVecs),axis=0))/np.std(np.asarray(self.__wordVecs),axis=0)
        
        
