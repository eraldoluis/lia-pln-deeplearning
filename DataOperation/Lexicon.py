#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''

class Lexicon:
    
    UNKNOWN_INDEX = 0
    UNKNOWN_VALUE = 'u*uu+n+kk*k'

    def __init__(self,filePath=''):
        self.__lexicon = []
        self.__lexiconDict = {}
        
        self.put(Lexicon.UNKNOWN_VALUE)
        
        if filePath:
            self.putUsingFile(filePath)

    def getLen(self):
        '''
        Return the number of words in the lexicon.
        '''
        return len(self.__lexicon)
    
    def putUsingFile(self,vocabfilename):
        # Read the vocabulary file (one word per line).
        fVoc = open(vocabfilename, 'r')
        for line in fVoc:
            word = line.strip()
            # Ignore empty lines.
            if len(word) == 0:
                continue
            self.put(word)
        fVoc.close()

    def put(self, word):
        '''
        Include a new word in the lexicon and return its index. If the word is
        already in the lexicon, then just return its index.
        '''
        word = word.lower()
        idx = self.__lexiconDict.get(word)
        if idx is None:
            # Insert a unseen word in the lexicon.
            idx = len(self.__lexicon)
            self.__lexicon.append(word)
            self.__lexiconDict[word] = idx
        
        return idx

    def getLexicon(self, index):
        '''
        Return the word in the lexicon that is stored in the given index.
        '''
        return self.__lexicon[index]

    def getLexiconIndex(self, word):
        '''
        Return the index of the given word. If the word is not in the lexicon,
        the return 0 (the unknown lexicon).
        '''
        word = word.lower()
        return self.__lexiconDict.get(word, self.UNKNOWN_INDEX)
    
    def isUnknownIndex(self,index):
        return index == self.UNKNOWN_INDEX