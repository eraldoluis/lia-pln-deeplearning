#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
import codecs

class Lexicon:

    def __init__(self,filePath=''):
        self.__lexicon = []
        self.__lexiconDict = {}
        self.unknown_index = -1
        
        if filePath:
            self.putUsingFile(filePath)
    
    def getLexiconDict(self):
        return self.__lexiconDict

    def getLen(self):
        '''
        Return the number of words in the lexicon.
        '''
        return len(self.__lexicon)
    
    def putUsingFile(self,vocabfilename):
        # Read the vocabulary file (one word per line).
        fVoc = codecs.open(vocabfilename, 'r', 'utf-8')
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
        return self.__lexiconDict.get(word, self.unknown_index)
      
    def getDict(self):
        return self.__lexiconDict
    
    def getUnknownIndex(self):
        return self.unknown_index
    
    def isUnknownIndex(self,index):
        return index == self.unknown_index
    
    def isWordExist(self,word):
        return not self.isUnknownIndex(self.getLexiconIndex(word))
    
    def setUnknownIndex(self,unknown_index):
        self.unknown_index = unknown_index
        
    def printi(self):
        print self.__lexicon
        print self.__lexiconDict
        print self.unknown_index
