#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
from DataOperation.Lexicon import Lexicon


class FeatureFactory:
 
    def readData(self, filename,lexicon,lexiconOfLabel, wordVecs, addWordUnkown=False):
        '''
        Read the data from a file and return a matrix which the first row is the words indexes  and second row is the labels values
        '''
        data = [[],[]]
        indexes = data[0]
        labels = data[1]

        f = open(filename, 'r')
        for line in f:
            line_split = line.split()
            # Ignore empty lines.
            if len(line_split) < 2:
                continue
            word = line_split[0]
            label = line_split[1]
            
            lexiconIndex = lexicon.getLexiconIndex(word)
            
            if addWordUnkown and lexicon.isUnknownIndex(lexiconIndex):
                lexiconIndex = lexicon.put(word)
                wordVecs.append(None)
            
            indexes.append(lexiconIndex)
            labels.append(lexiconOfLabel.put(label))
        
        f.close()

        return data

    def getLexiconLen(self):
        '''
        Return the number of words in the lexicon.
        '''
        return self.__lexicon.getLen()

    def getLexicon(self):
        return self.__lexicon
    
    def getNumberOfLabel(self):
        return self.__lexiconOfLabel.getLen()
   
    def getLexiconOfLabel(self):
        return self.__lexiconOfLabel

    def getWordVector(self):
        return self.__wordVecs