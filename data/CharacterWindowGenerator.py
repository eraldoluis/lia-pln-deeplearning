#!/usr/bin/env python
# -*- coding: utf-8 -*-


from data.Window import Window
from data.FeatureGenerator import FeatureGenerator


class CharacterWindowGenerator(FeatureGenerator):
    """
    Generate sequences of character windows for a given sequence of words. For each word in the given sequence, a
    sequence of character windows will be generated.

    This class considers that words has the same number of character (numMaxChar).
    If a word has less than numMaxChar characters, then it will be filled with an artificial character to become a
    numMaxChar-character word. If the word has more than numMaxChar characters, then it will be used only its numMaxChar
    last characters (suffix).
    """

    def __init__(self, lexicon, numMaxChar, charWindowSize, wrdWindowSize, artificialChar, startPadding,
                 endPadding=None, startPaddingWrd=None, endPaddingWrd=None, filters=[]):
        """
        Create a character window feature generator.

        TODO: Irving, comentar cada parâmetro.

        :param lexicon:
        :param numMaxChar:
        :param charWindowSize:
        :param wrdWindowSize:
        :param artificialChar:
        :param startPadding:
        :param endPadding:
        :param startPaddingWrd:
        :param endPaddingWrd:
        :param filters:
        """
        self.__charWindowSize = charWindowSize
        self.__charWindow = Window(lexicon, charWindowSize, startPadding, endPadding)
        self.__wordWindow = Window(lexicon, wrdWindowSize, startPaddingWrd, endPaddingWrd)
        self.__lexicon = lexicon
        self.__numMaxChar = numMaxChar
        self.__wrdWindowSize = wrdWindowSize
        self.__filters = filters
        self.__artificialCharWindow = [lexicon.getLexiconIndex(artificialChar)] * charWindowSize

    def generate(self, sequence):
        """
        Generate character-level features for the given sequence of tokens.

        :type sequence: list
        :param sequence: sequence of tokens
        :return:
        """
        wordWindowList = self.__wordWindow.buildWindows(sequence)
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

                    for f in self.__filters:
                        word = f.filter(word, sequence)

                    allCharacters = [c for c in word]
                    lenWord = len(word)
                    chardIdxs = []

                    for c in allCharacters:
                        chardIdxs.append(self.__lexicon.put(c))

                    charWindowList = list(self.__charWindow.buildWindows(chardIdxs))

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