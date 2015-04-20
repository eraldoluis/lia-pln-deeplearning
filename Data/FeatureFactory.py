#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
from Data.WordVector import WordVector
from Data.Lexicon import Lexicon


class FeatureFactory:

    def __init__(self,wordVectorSize):
        self.__lexicon = Lexicon(); 
        self.__lexiconOfLabel = Lexicon();
        self.__wordVecs = WordVector(wordVectorSize);
        
    def readWordVectors(self, vecfilename, vocabfilename):
        '''
        Load a dictionary from vocabfilename and the vector (embedding)
        for each word from vecfilename.
        '''
        # Read the vocabulary file (one word per line).
        fVoc = open(vocabfilename, 'r')
        for line in fVoc:
            word = line.strip()
            # Ignore empty lines.
            if len(word) == 0:
                continue
            self.__lexicon.put(word)
        fVoc.close()

        # Read the vector file (one word vector per line).
        # Each vector represents a word from the vocabulary.
        # The vectors must be in the same order as in the vocabulary.
        fVec = open(vecfilename, 'r')
        for line in fVec:
            self.__wordVecs.append([float(num) for num in line.split()])
        fVec.close()
        
        self.__wordVecs.setLenWordVectorAuto()

        assert (self.__wordVecs.getLength() == self.__lexicon.getLen())

    def readData(self, filename,addWordUnkown=False):
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
            
            lexiconIndex = self.__lexicon.getLexiconIndex(word)
            
            if addWordUnkown and lexiconIndex == 0:
                lexiconIndex = self.__lexicon.put(word)
                self.__wordVecs.append(None)
            
            indexes.append(lexiconIndex)
            labels.append(self.__lexiconOfLabel.put(label))
        
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