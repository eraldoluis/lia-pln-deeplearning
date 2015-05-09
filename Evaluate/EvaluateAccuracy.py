#!/usr/bin/env python
# -*- coding: utf-8 -*-

from timeit import itertools


class EvaluateAccuracy:
    
    def evaluate(self,predicts,corrects):
            
        if len(predicts) != len(corrects):
            raise Exception('O número de predições é diferente do número de exemplos')
        
        sum = 0.
        
        for p,c in itertools.izip(p,c):
            if p == c:
                sum +=1.
        
        
        return sum / len(predicts)