#!/usr/bin/env python
# -*- coding: utf-8 -*-

from timeit import itertools


class EvaluateAccuracy:
    
    def evaluate(self,predicts,corrects):
            
        if len(predicts) != len(corrects):
            raise Exception('O número de predições é diferente do número de exemplos')
        
        sum = 0.
        total = 0.0
        
        for predSentence,cSentence in itertools.izip(predicts,corrects):
            for p,c in itertools.izip(predSentence,cSentence):
                if p == c:
                    sum +=1.
                
                total += 1
        
        return sum / total