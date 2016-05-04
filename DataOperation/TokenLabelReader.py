#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
import re
import codecs
import logging
from DataOperation.TransformNumberToZeroFilter import TransformNumberToZeroFilter
from DataOperation.TransformLowerCaseFilter import TransformLowerCaseFilter

class TokenLabelReader:
    
    def __init__(self, fileWithFeatures, tokenLabelSeparator, task):
        self.fileWithFeatures = fileWithFeatures
        self.__tokenLabelSeparator = tokenLabelSeparator
        self.task = task
    
    def readTestData(self, filename, lexicon, lexiconOfLabel, lexiconRaw, 
                     separateSentences=True, addWordUnknown=False, 
                     withCharwnn=False, charVars=[None, None, {}, []], 
                     addCharUnknown=False, filters=[], unknownDataTest=None, 
                     unknownDataTestCharIdxs=None):
        return self.readData(filename, lexicon, lexiconOfLabel, lexiconRaw, 
                             None, separateSentences, False, withCharwnn, 
                             charVars, False, filters, None, unknownDataTest, 
                             unknownDataTestCharIdxs)
    
    def addToken(self, lexicon, wordVecs, addWordUnknown, filters, 
                 indexesBySentence, rawWord, setWordsInDataSet, unknownData, 
                 lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence, 
                 withCharwnn, charVars, addCharUnknown, unknownDataCharIdxs):
        
        charcon = charVars[0]
        charVector = charVars[1]
        charIndexesOfLexiconRaw = charVars[2]
        numCharsOfLexiconRaw = charVars[3]
        
        word = rawWord
        
        for f in filters:
            if isinstance(f, TransformLowerCaseFilter):
                word = f.filter(word)
            elif isinstance(f, TransformNumberToZeroFilter):
                word = f.filter(word) 
            else:
                rawWord = f.filter(rawWord)
                word = f.filter(word)
                
        #word = filters[-1].filter(rawWord)        
        lexiconIndex = lexicon.getLexiconIndex(word)
        

        if addWordUnknown and lexicon.isUnknownIndex(lexiconIndex):
            lexiconIndex = lexicon.put(word)
            if isinstance(wordVecs, list):
                for wv in wordVecs:
                    wv.append(None)
            else:
                wordVecs.append(None)
        
        if setWordsInDataSet is not None:
            setWordsInDataSet.add(word)
            
        if lexicon.isUnknownIndex(lexiconIndex) and unknownData is not None:
            unknownData.append(word)
            
        if lexiconIndex < 0:
            raise Exception(u'The lexiconIndex of ' + word + u' is negative. Probably the unknown word was not set');
        
        indexesBySentence.append(lexiconIndex)
        
        if withCharwnn:
            
            lexiconIndex = lexiconRaw.getLexiconIndex(rawWord)
            if lexiconRaw.isUnknownIndex(lexiconIndex):
                lexiconIndex = lexiconRaw.put(rawWord)
                numCharsOfLexiconRaw.append(len(rawWord))
                            
                charIndexes = []
                for char in rawWord:
                    idx = charcon.getLexiconIndex(char)
                    if addCharUnknown and charcon.isUnknownIndex(idx):
                        idx = charcon.put(char)
                        charVector.append(None)
                    charIndexes.append(idx)
                charIndexesOfLexiconRaw[lexiconIndex] = charIndexes
                if unknownDataCharIdxs is not None:
                    unknownDataCharIdxs.append(charIndexes) 
                    
            indexesOfRawBySentence.append(lexiconIndex)
            numCharsOfRawBySentence.append(len(rawWord))        
            
        

    def addLabel(self, lexiconOfLabel, labelsBySentence, token):
        labelsBySentence.append(lexiconOfLabel.put(token))
                                        
    def readTokenAndLabelOfFileWithFeature(self, lexicon, lexiconOfLabel, wordVecs, addWordUnknown, filters, indexesBySentence, labelsBySentence, lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence , token, setWordsInDataSet, unknownData, withCharwnn, charVars, addCharUnknown, unknownDataCharIdxs):
                
        prefWord = 'word='
        
        if prefWord in token:
            word = token[len(prefWord):]
            self.addToken(lexicon, wordVecs, addWordUnknown, filters, indexesBySentence, word, setWordsInDataSet, unknownData,
                          lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence, withCharwnn, charVars, addCharUnknown, unknownDataCharIdxs)
        elif re.search(r'^([A-Z]|\W)', token) is not None:
            self.addLabel(lexiconOfLabel, labelsBySentence, token)
                                
    def readTokenAndLabelOfRawFile(self, lexicon, lexiconOfLabel, wordVecs, addWordUnknown, filters, indexesBySentence, labelsBySentence, lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence, token, setWordsInDataSet, unknownData, withCharwnn, charVars, addCharUnknown, unknownDataCharIdxs):
    
        s = token.rsplit(self.__tokenLabelSeparator, 1)
        
        if len(s[1]) == 0:
            logging.getLogger("Logger").warn("It was not found the label from "\
                         "the token " + token + ". We give to this token "\
                         " a label equal to"\
                         " the tokenLabelSeparator( " + self.__tokenLabelSeparator + ")")
              
            s[1] = self.__tokenLabelSeparator
                
        if (len(s[0]) and len(s[1])):
            self.addToken(lexicon, wordVecs, addWordUnknown, filters, indexesBySentence, s[0], setWordsInDataSet, unknownData,
                          lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence, withCharwnn, charVars, addCharUnknown, unknownDataCharIdxs)
        
            self.addLabel(lexiconOfLabel, labelsBySentence, s[1])
    
    def readTokenAndSentOfRawFile(self, lexicon, wordVecs, addWordUnknown, filters, indexesBySentence, lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence, token, setWordsInDataSet, unknownData, withCharwnn, charVars, addCharUnknown, unknownDataCharIdxs):
    
        self.addToken(lexicon, wordVecs, addWordUnknown, filters, indexesBySentence, token, setWordsInDataSet, unknownData,
                          lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence, withCharwnn, charVars, addCharUnknown, unknownDataCharIdxs)
        
 
    def readData(self, filename, lexicon, lexiconOfLabel, lexiconRaw, wordVecs=None, separateSentences=True, addWordUnknown=False, withCharwnn=False, charVars=[None, None, {}, []], addCharUnknown=False, filters=[], setWordsInDataSet=None, unknownDataTest=[], unknownDataTestCharIdxs=None):
        
        if self.task == 'postag':
            return self.readDataPostag(filename, lexicon, lexiconOfLabel, lexiconRaw, wordVecs, separateSentences, addWordUnknown, withCharwnn, charVars, addCharUnknown, filters, setWordsInDataSet, unknownDataTest, unknownDataTestCharIdxs)
        elif self.task == 'sentiment_analysis':
            return self.readDataSentAnalysis(filename, lexicon, lexiconOfLabel, lexiconRaw, wordVecs, separateSentences, addWordUnknown, withCharwnn, charVars, addCharUnknown, filters, setWordsInDataSet, unknownDataTest, unknownDataTestCharIdxs)
            
    
    def readDataPostag(self, filename, lexicon, lexiconOfLabel, lexiconRaw, wordVecs=None, separateSentences=True, addWordUnknown=False, withCharwnn=False, charVars=[None, None, {}, []], addCharUnknown=False, filters=[], setWordsInDataSet=None, unknownDataTest=[], unknownDataTestCharIdxs=None):
                
        '''
        Read the data from a file and return a matrix which the first row is the words indexes, second row is the labels values and 
        the third row has the indexes of the raw words, and the fourth the number of chars each word in the training set has.
        '''
        data = [[], [], [], []]
        indexes = data[0]
        labels = data[1]
        indexesOfRaw = data[2]
        numCharsOfRaw = data[3]
        
        
        
        func = self.readTokenAndLabelOfFileWithFeature if self.fileWithFeatures else self.readTokenAndLabelOfRawFile
        
        f = codecs.open(filename, 'r', 'utf-8')
        
        
        for line in f:
            
            line_split = line.split()
            # Ignore empty lines.
            if len(line_split) == 0:
                continue
            
            if separateSentences:
                indexesBySentence = []
                labelsBySentence = []
                indexesOfRawBySentence = []
                numCharsOfRawBySentence = []
                unknownDataBySentence = [] if unknownDataTest is not None else None
            else:
                indexesBySentence = indexes
                indexesOfRawBySentence = indexesOfRaw
                numCharsOfRawBySentence = numCharsOfRaw
                labelsBySentence = labels
                unknownDataBySentence = unknownDataTest
            
            for token in line_split:
                func(lexicon, lexiconOfLabel, wordVecs, addWordUnknown, filters, indexesBySentence, labelsBySentence,
                      lexiconRaw, indexesOfRawBySentence, numCharsOfRawBySentence, token, setWordsInDataSet, unknownDataBySentence, withCharwnn, charVars, addCharUnknown, unknownDataTestCharIdxs)
            
            if len(indexesBySentence) != len(labelsBySentence):
                raise Exception('Número de tokens e labels não são iguais na linha: ' + line)
                         
            if separateSentences:
                indexes.append(indexesBySentence)
                labels.append(labelsBySentence)
                indexesOfRaw.append(indexesOfRawBySentence)
                numCharsOfRaw.append(numCharsOfRawBySentence)
                
                if unknownDataTest is not None:
                    unknownDataTest.append(unknownDataBySentence)
                
        f.close()
        
        
        return data
    
    def readDataSentAnalysis(self, filename, lexicon, lexiconOfLabel, lexiconRaw, wordVecs=None, separateSentences=True, addWordUnknown=False, withCharwnn=False, charVars=[None, None, {}, []], addCharUnknown=False, filters=[], setWordsInDataSet=None, unknownDataTest=[], unknownDataTestCharIdxs=None):
                
        '''
        Read the data from a file and return a matrix which the first row is the words indexes, second row is the labels values and 
        the third row has the indexes of the raw words, and the fourth the number of chars each word in the training set has.
        '''
        data = [[], [], [], []]
        indexes = data[0]
        labels = data[1]
        indexesOfRaw = data[2]
        numCharsOfRaw = data[3]
        
                
        func = self.readTokenAndSentOfRawFile
        
        f = codecs.open(filename, 'r', 'utf-8')
        
        for line in f:
            
            row = line.split(',',1)
            
            if len(row) < 2:
                continue
                
            line_split = row[1].split()
            
            # Ignore empty twittes.
            if len(line_split) == 0:
                continue
                
            labels.append(lexiconOfLabel.put(row[0]))
                               
            indexesBySentence = []
            indexesOfRawBySentence = []
            numCharsOfRawBySentence = []
            unknownDataBySentence = [] if unknownDataTest is not None else None
                
            for token in line_split:
                func(lexicon, wordVecs, addWordUnknown, filters, indexesBySentence, lexiconRaw, 
                     indexesOfRawBySentence, numCharsOfRawBySentence, token, setWordsInDataSet,
                     unknownDataBySentence, withCharwnn, charVars, addCharUnknown, unknownDataTestCharIdxs)
                
            indexes.append(indexesBySentence)
            indexesOfRaw.append(indexesOfRawBySentence)
            numCharsOfRaw.append(numCharsOfRawBySentence)
                
            if unknownDataTest is not None:
                unknownDataTest.append(unknownDataBySentence)
                
        f.close()
                            
        return data
    
    
