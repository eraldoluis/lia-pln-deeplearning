#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import random
import os

class GenerateFolds:    
   
    def readData(self, newdirname,k, labels, line):
        os.mkdir(newdirname)
        
        folds = []
        for i in range (k):
            folds.append([])
         
        for label, linesOfLabel in zip (labels, line) :
            tam = len(linesOfLabel)
            idx = range(tam)
            random.shuffle(idx)
            
            j = 0
            for i in range(tam/k):
                for fold in folds:
                    fold.append(label+','+linesOfLabel[idx[j]])
                    j+=1
            for fold in folds:
                if j < tam:
                    fold.append(label+','+linesOfLabel[idx[j]])
                    j+=1
                else:
                    break
                
        for i in range(k):
            fileoutTrain = newdirname + 'train_'+ str(i+1) + '.txt'
            fileoutTest =  newdirname + 'test_'+ str(i+1) + '.txt'
            
            fTrain = codecs.open(fileoutTrain, 'w','utf-8')
            fTest  = codecs.open(fileoutTest , 'w','utf-8')
            
            for j in range(k):
                if j == i:
                    for line in folds[j]:
                        fTest.write(line)
                else:
                    for line in folds[j]:
                        fTrain.write(line)
                        
            fTrain.close()
            fTest.close() 
      
