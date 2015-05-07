#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
import re
import time


class MacMorphoReader:    
        
    def readData(self, filename,lexicon,lexiconOfLabel, wordVecs, separateSentences= True, addWordUnkown=False):
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
            a +=1
            
            if a == 1000:
                break
            
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

        return data
 