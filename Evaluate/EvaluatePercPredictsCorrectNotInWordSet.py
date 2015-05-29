#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import itertools
from numpy import ndarray


class EvaluatePercPredictsCorrectNotInWordSet:
    
    def __init__(self,lexicon,wordSet,testName):
        self.lexicon = lexicon
        self.set = wordSet
        self.testName = testName
    
    
    def evaluateWithPrint(self,predicts,corrects,inputData,unkwonDataTestVec):
        print "Result " + str(self.testName) + ": " + str(self.evaluate(predicts,corrects,inputData,unkwonDataTestVec))
        
    def evaluate(self,predicts,corrects,inputData,unkwonDataTestVec):
        predict = numpy.asarray(predicts)
        correct = numpy.asarray(corrects)
        inputData = numpy.asarray(inputData)
        
        if predict.shape != correct.shape:
            raise Exception('O número de predições é diferente do número de exemplos')
        
        total = 0
        sum = 0
        i = 0
        
        indexUnkownDataVec = 0
        
        while i < len(predict):
            if isinstance(predict[i], list) or isinstance(predict[i], ndarray) : 
                
                if len(predict[i]) != len(correct[i]):
                    raise Exception('O número de predições é diferente do número de exemplos')
                
                indexUnkownDataVec = 0
                       
                for p,c,input in itertools.izip(predict[i],correct[i],inputData[i]):
                    
                    if self.lexicon.isUnknownIndex(input):
                        word = unkwonDataTestVec[i][indexUnkownDataVec]
                        indexUnkownDataVec += 1
                    else:
                        word = self.lexicon.getLexicon(input)
                    
                    if word in self.set:
                        continue;
                    
                    if p == c:
                        sum += 1
                    total += 1
            else:
                
                if self.lexicon.isUnknownIndex(inputData[i]):
                    word = unkwonDataTestVec[indexUnkownDataVec]
                    indexUnkownDataVec += 1
                else:
                    word = self.lexicon.getLexicon(inputData[i])
                
                if word  not in self.set:
                    if predict[i] == correct[i]:
                        sum += 1
                    
                    total += 1
                    
            i+=1
            
        if total == 0:
            return NaN
        
        return sum / float(total)