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
    
    
    def evaluateWithPrint(self,predicts,corrects,inputData):
        print "Result " + str(self.testName) + ": " + str(self.evaluate(predicts,corrects,inputData))
        
    def evaluate(self,predicts,corrects,inputData):
        predict = numpy.asarray(predicts)
        correct = numpy.asarray(corrects)
        inputData = numpy.asarray(inputData)
        
        if predict.shape != correct.shape:
            raise Exception('O número de predições é diferente do número de exemplos')
        
        total = 0
        sum = 0
        
        i = 0
        
        while i < len(predict):
            if isinstance(predict[i], list) or isinstance(predict[i], ndarray) : 
                
                if len(predict[i]) != len(correct[i]):
                    raise Exception('O número de predições é diferente do número de exemplos')
                       
                for p,c,input in itertools.izip(predict[i],correct[i],inputData[i]):
                    if self.lexicon.getLexicon(input) in self.set:
                        continue;
                    
                    if p == c:
                        sum += 1
                    total += 1
            else:
                if self.lexicon.getLexicon(inputData[i]) not in self.set:
                    if predict[i] == correct[i]:
                        sum += 1
                    
                    total += 1
                    
            i+=1
            
        if total == 0:
            return NaN
        
        return sum / float(total)