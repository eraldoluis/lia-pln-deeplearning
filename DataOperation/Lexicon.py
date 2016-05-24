#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
import codecs


class Lexicon(object):
    """
    Represents a lexicon of words.
    Each added word in this lexicon will be represented by a integer.
    This class has a variable '__stopAdd' that control if the lexicon will or not insert new words.
    If a word in new to lexicon and '__stopAdd' is true, so this lexicon will return a index which
        is related with all unknown words.
    This special index need to be set with setUnknownIndex.
    """

    def __init__(self):
        self.__lexicon = []
        self.__lexiconDict = {}
        self.unknown_index = -1
        self.__stopAdd = False

    def isStopped(self):
        """
        :return: return if lexicon is adding new words
        """
        return self.__stopAdd

    def getLexiconDict(self):
        """
        :return: return a dictionary which contains the integers which represent each word.
        """
        return self.__lexiconDict

    def getLen(self):
        '''
        Return the number of words in the lexicon.
        '''
        return len(self.__lexicon)

    def put(self, word):
        '''
        Include a new word in the lexicon and return its index. If the word is
        already in the lexicon, then just return its index.
        If a word in new to lexicon and '__stopAdd' is true, so this lexicon will return a index which
        is related with all unknown words.
        '''

        idx = self.__lexiconDict.get(word)

        if idx is None:
            if self.isStopped():
                return self.getUnknownIndex()

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
            so returns the unknown index.
        '''
        return self.__lexiconDict.get(word, self.unknown_index)

    def getUnknownIndex(self):
        return self.unknown_index

    def isUnknownIndex(self, index):
        return index == self.unknown_index

    def exist(self, word):
        return not self.isUnknownIndex(self.getLexiconIndex(word))

    def setUnknownIndex(self, unknown_index):
        self.unknown_index = unknown_index

    def stopAdd(self):
        """
        Tell the class to stop adding new words

        :return:
        """
        self.__stopAdd = True