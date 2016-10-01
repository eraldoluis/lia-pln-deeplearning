#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Filters are classes that change the token, like transform to lower case the letters 
'''

import re


class Filter:
    def filter(self, token, sentence):
        raise NotImplementedError()


class TransformLowerCaseFilter:
    def filter(self, token, sentence):
        return token.lower()





class TransformNumberToZeroFilter:
    def filter(self, token, sentence):
        return re.sub('[0-9]', '0', token)

class NeutralQuotesFilter:
    """
    Transform neutral quotes("'`) to opening(``) or closing quotes('')
    """

    def __init__(self):
        self.__lastSentence = ""
        self.__isNextQuoteOpen = True


    def filter(self, token, sentence):
        if re.search(r"^[\"`']$",token):
            if self.__lastSentence == sentence:
                if self.__isNextQuoteOpen:
                    self.__isNextQuoteOpen = False
                    return "``"
                else:
                    self.__isNextQuoteOpen = True
                    return "''"
            else:
                self.__lastSentence = sentence
                self.__isNextQuoteOpen = False
                return "``"


        return token


class UrlFilter:
    """
    Tokens starting with “www.”, “http.” or ending with “.org”, “.com” e ".net" are converted to a “#URL” symbol
    """
    def filter(self, token, sentence):
        token = re.sub(r"^((https?:\/\/)|(www\.))[^\s]+$", "#URL", token )
        token = re.sub(r"^[^\s]+(\.com|\.net|\.org)\b([-a-zA-Z0-9@;:%_\+.~#?&//=]*)$", "#URL", token)

        return token

class RepeatedPunctuationFilter:
    """
    Repeated punctuations such as “!!!!” are collapsed into one.
    """
    def filter(self, token, sentence):
        token = re.sub(r"^[,:;><!?_\//]{2,}$", token[0], token )
        token = re.sub(r"^[.]{4,}$", "...", token)
        token = re.sub(r"^[.]{2,2}$", ".", token)
        token = re.sub(r"^[--]{3,}$", "--", token)

        return token


class LeftRightBracketsFilter:
    """
    Left brackets such as “<”,“{” and “[” are converted to “-LRB-”. Similarly, right brackets are converted to “-RRB-”
    """

    def filter(self, token, sentence):
        token = re.sub('[<{[]', '-LRB-', token)
        token = re.sub('[]>}]', '-RRB-', token)

        return token

class UpperCaseWordFilter:
    """
    Left brackets such as “<”,“{” and “[” are converted to “-LRB-”. Similarly, right brackets are converted to “-RRB-”
    """

    def filter(self, token, sentence):

        if len(token) > 4  and token.isUpper():
            return token.lower()

        return token


class WordWithDigitsFilter:
    """
    Consecutive occurrences of one or more digits within a word are replaced with “#DIG”
    """

    def filter(self, token, sentence):

        if re.search("[0-9]+", token) and re.search("[^\s.,/:0-9-]", token):
            return "#DIG"

        return token

class LowerCaseSANCLFilter:
    """
    This filter don't transform to lower case these specials symbols: -RRB,-LRB, #DIG, #URL
    """

    def filter(self, token, sentence):
        if re.search("^(-RRB-)|(-LRB-)|(#DIG)|(#URL)$", token):
            return token

        return token.lower()

