#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
import re
import time


class MacMorphoReader:    
    def readTestData(self, filename,lexicon,lexiconOfLabel,separateSentences=True,filters=[]):
        return self.readData(filename, lexicon, lexiconOfLabel,None,separateSentences,False,filters)
    
    def readData(self, filename,lexicon,lexiconOfLabel, wordVecs=None, separateSentences= True, addWordUnkown=False,filters=[]):
        '''
        Read the data from a file and return a matrix which the first row is the words indexes  and second row is the labels values
        '''
        data = [[],[]]
        indexes = data[0]
        labels = data[1]
        
        f = open(filename, 'r')
        a = 0
        prefWord = 'word='
        
        for line in f:
#             a +=1
#                  
#             if a ==10:
#                 break;
            
            line_split = line.split()
            # Ignore empty lines.
            if len(line_split) < 2:
                continue
            
            if separateSentences:
                indexesBySentence = []
                labelsBySentence = []
            else:
                indexesBySentence = indexes
                labelsBySentence = labels
            
            for token in line_split:
                
                if prefWord in token:
                    word = token[len(prefWord):]
                    
                    for filter in filters:
                        word = filter.filter(word);
                    
                    lexiconIndex = lexicon.getLexiconIndex(word)
                    
                    if addWordUnkown and lexicon.isUnknownIndex(lexiconIndex):
                        lexiconIndex = lexicon.put(word)
                        wordVecs.append(None)
                    
                    indexesBySentence.append(lexiconIndex)
                elif re.search(r'^([A-Z])', token) is not None:
                    labelsBySentence.append(lexiconOfLabel.put(token))
            
            if separateSentences:
                indexes.append(indexesBySentence)
                labels.append(labelsBySentence)
                
        f.close()
        
#         list = []
#         i = 0
#         for l in data[0]:
#             if len(l) < 5:
#                 list.append((i,len(l)))
#             i+=1
# 
#         print list
        return data
 