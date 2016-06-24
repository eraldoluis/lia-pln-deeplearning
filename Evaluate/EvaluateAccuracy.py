#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
from itertools import chain
import itertools
from numpy import ndarray

class EvaluateAccuracy:
    
    def stardizeVector(self,v):
        v = numpy.asarray(v)
        
        if v.ndim == 1:
            return v
        
        return numpy.fromiter(chain.from_iterable(v))
    
    def evaluateWithPrint(self,predicts,corrects):
        print "Accuracy Test:  ",self.evaluate(predicts,corrects)
        
    def evaluate(self,predicts,corrects):
        predict = numpy.asarray(predicts)
        correct = numpy.asarray(corrects)
        
        if predict.shape != correct.shape:
            raise Exception('O número de predições é diferente do número de exemplos')
        
        total = 0
        sum = 0
        
        i = 0
        
        while i < len(predict):
            if isinstance(predict[i], list) or isinstance(predict[i], ndarray) : 
                
                if len(predict[i]) != len(correct[i]):
                    raise Exception('O número de predições é diferente do número de exemplos')
                       
                for p,c in itertools.izip(predict[i],correct[i]):
                    
                    if p == c:
                        sum += 1
                    total += 1
            else:
                
                if predict[i] == correct[i]:
                    sum += 1
                
                total += 1
            
            i+=1
        
        return sum / float(total)