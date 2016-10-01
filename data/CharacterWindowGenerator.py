#!/usr/bin/env python
# -*- coding: utf-8 -*-


from data.Context import Window
from data.FeatureGenerator import FeatureGenerator


class CharacterWindowGenerator(FeatureGenerator):
    """
    Generate the character window of the words.
    This class considers that words has the same number of character, that will call numMaxChar.
    If the word has a number of characters lesser than numChar,
        so this class will fill this word with empty character.
    If the word has a number of characters greater than numChar,
        so will thrown way the begin of the word.
    """

    def __init__(self, embedding, numMaxChar, charWindowSize, wrdWindowSize, artificialChar, startPadding,
                 endPadding=None, startPaddingWrd=None, endPaddingWrd=None):
        self.__charWindowSize = charWindowSize
        self.__charWindow = Window(charWindowSize)
        self.__wordWindow = Window(wrdWindowSize)
        self.__embedding = embedding
        self.__numMaxChar = numMaxChar
        self.__wrdWindowSize = wrdWindowSize
        self.__artificialCharWindow = [embedding.getLexiconIndex(artificialChar)] * charWindowSize

        self.__startPaddingIdx, self.__endPaddingIdx = Window.checkPadding(startPadding, endPadding, embedding)

        if not (startPaddingWrd or startPaddingWrd):
            startPaddingWrd = "PADD_WRD"

        self.__startPaddingWrdIdx, self.__endPaddingWrdIdx = Window.checkPadding(startPaddingWrd, endPaddingWrd,
                                                                                 embedding)

    def generate(self, rawData):
        wordWindowList = self.__wordWindow.buildWindows(rawData, self.__startPaddingWrdIdx,
                                                        self.__endPaddingWrdIdx)
        charsAllExamples = []
        for wordWindow in wordWindowList:
            charsOfExample = []

            for word in wordWindow:
                if isinstance(word,int):
                    # If the word is a integer so it is a padding of the window word.
                    # In this case, we create a window formed by this integer.
                    # This integer is the index of character embedding that represents the padding of word window.
                    paddingWindow = [word] * self.__charWindowSize
                    charWindowList = [paddingWindow]
                    lenWord = 1
                else:
                    allCharacters = [c for c in word]
                    lenWord = len(word)
                    chardIdxs = []

                    for c in allCharacters:
                        chardIdxs.append(self.__embedding.put(c))

                    charWindowList = list(self.__charWindow.buildWindows(chardIdxs, self.__startPaddingIdx,self.__endPaddingIdx))

                if lenWord >= self.__numMaxChar:
                    # Get only the numMaxCh-character long suffix of the word.
                    charWindowList = charWindowList[-self.__numMaxChar:]
                else:
                    # Get the whole word and append artificial characters to
                    # fill it up to numMaxCh characters.
                    charWindowList = charWindowList + [self.__artificialCharWindow] * (self.__numMaxChar - lenWord)

                charsOfExample.append(charWindowList)

            charsAllExamples.append(charsOfExample)

        return charsAllExamples