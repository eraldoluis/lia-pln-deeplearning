#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from nnet.LinearLayer import LinearLayer
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.Util import LearningRateUpdNormalStrategy
from nnet.Layer import Layer

class EmbeddingConvolutionalLayer(Layer):
    """
    Convolutional layer of embedding features.
    
    The input of this layer is a 4-D tensor whose shape is:
        (numExs, szWrdWin, numMaxCh, szChWin)
    where
        numExs is the number of examples in a training (mini) batch,
        szWrdWin is the size of the word window
            (the convolution is independently performed for each index in this dimension),
        numMaxCh is the number of characters used to represent words
            (the convolution is performed over this dimension),
        szChWin is the size of the character window
            (each input for the convolution filters is composed by this number of features).
    
    The value numMaxCh, the number of characters used to represent a word,
    is fixed for all word to speedup training. For words that are shorter
    than this value, we extend them with an artificial character. For words
    that are longer than this value, we use only the last numMaxCh
    characters in them.
    
    Thus, this layer is not really convolutional (with variable-sized inputs),
    but it is sufficient for many applications and is much faster than an
    ordinary convolutional layer.
    """
    
    def __init__(self, charcon, charVectors, charIdxWord, numCharsOfWord,
                 charWindowSize, wordWindowSize , convSize,
                 numClasses, c=0.0,
                 learningRateUpdStrategy=LearningRateUpdNormalStrategy(),
                 separateSentence=False, withAct=False,
                 charVecsUpdStrategy='normal', charAct="tanh", norm_coef=1.0,
                 structGrad=True):
        self.CharIdxWord = charIdxWord
        
        # Feature embedding
        self.__embedding = theano.shared(name='conv_embedding',
                                         value=np.asarray(charVectors.getWordVectors(),
                                                          dtype=theano.config.floatX),
                                         borrow=True)
        
        self.numChars = len(charVectors.getWordVectors())
        self.charSize = charVectors.getLenWordVector()
        self.charWindowSize = charWindowSize
        self.wordWindowSize = wordWindowSize
        self.startSymbol = charcon.getLexiconIndex(WindowModelBasic.startSymbolStr)
        self.endSymbol = charcon.getLexiconIndex(WindowModelBasic.endSymbolStr)
        self.numClasses = numClasses
        self.separateSentence = separateSentence
        
        self.output = None
        self.withAct = withAct
        self.charVecsUpdStrategy = charVecsUpdStrategy
        
        self.regularizationFactor = theano.shared(c)
        self.convSize = convSize
        
        # This is the fixed size of all words.
        self.maxLenWord = numCharsOfWord
        
        self.charAct = charAct
        self.norm_coef = norm_coef
        
        self.__structGrad = structGrad
        
        # Input variable for this layer. Its shape is (numExs, szWrdWin, numMaxCh, szChWin)
        # where numExs is the number of examples in the training batch,
        #       szWrdWin is the size of the word window,
        #       numMaxCh is the number of characters used to represent words, and
        #       szChWin is the size of the character window.
        self.charWindowIdxs = T.itensor4(name="charWindowIdxs")
        
        # We use the symbolic shape of the input to perform all dimension
        # transformations (reshape) necessary for the computation of this layer.
        shape = T.shape(self.charWindowIdxs)
        numExs = shape[0]
        szWrdWin = shape[1]
        numMaxCh = shape[2]
        szChWin = shape [3]
        
        # Character embedding layer.
        self.__embedLayer = EmbeddingLayer(examples=self.charWindowIdxs.flatten(2),
                                           embedding=self.__embedding,
                                           structGrad=structGrad)
        
        # Size of the feature embedding.
        szChEmb = T.shape(self.__embedding)[1]
        
        # This is the bank of filters. It is an ordinary hidden layer.
        act = self.charAct if self.withAct else None
        hidInput = self.__embedLayer.getOutput().reshape((numExs * szWrdWin * numMaxCh, szChWin * szChEmb))
        self.hiddenLayer = LinearLayer(hidInput, self.charWindowSize * self.charSize, self.convSize, activation=act)
        
        # 3-D tensor with shape (numExs * szWrdWin, numMaxCh, numChFltrs).
        # This tensor is used to perform the max pooling along its 2nd dimension.
        o = self.hiddenLayer.getOutput().reshape((numExs * szWrdWin, numMaxCh, convSize))
        
        # Max pooling layer. Perform a max op along the character dimension.
        # The shape of the output is equal to (numExs*szWrdWin, convSize).
        m = T.max(o, axis=1)
        
        # The output is a 2-D tensor with shape (numExs, szWrdWin * numChFltrs).
        self.output = m.reshape((numExs, szWrdWin * convSize))
        
        # Montar uma lista com as janelas de char de cada palavra do dicionario
        self.AllCharWindowIndexes = self.getAllCharIndexes(charIdxWord)
        
    # Esta função retorna o índice das janelas dos caracteres de todas as palavras 
    def getAllWordCharWindowIndexes(self, inputData):
        
        if self.separateSentence:
            return self.getAllWordCharWindowIndexesBySentence(inputData)
        
        return self.getAllWordCharWindowIndexesByWord(inputData)

    def getAllWordCharWindowIndexesByWord(self, inputData):
        """
        Generate the character-level representation of each word.
        This representation is a 4-D array with shape equal to:
            (numTokens, szWrdWin, numMaxCh, szChWin)
        where numTokens is the number of tokens (words) in the training data,
        szWrdWin is the size of the word window, numMaxCh is the number
        of characters used to represent each word, and szChWin is the size
        of the character window.
        
        The value numMaxCh, the number of characters used to represent a word,
        is fixed for all word to speedup training. For words that are shorter
        than this value, we extend them with an artificial character. For words
        that are longer than this value, we use only the last numMaxCh
        characters in them.
        """
        chars = []
        
        # The artificial character is the last character of our dictionary.
        artificialChar = self.numChars - 1
        # The artificial character window that is appended to words shorter than numMaxCh.
        artCharWindow = [artificialChar] * self.charWindowSize

        # Number of characters used to represent each word.
        numMaxCh = self.maxLenWord
        
        for idxExample in xrange(len(inputData)):

            # Indexes of wordWindow within the window of the current word (idxExample).
            wordWindow = self.getWindowIndexes(idxExample, inputData)

            charsOfExample = []

            for w in wordWindow:
                # The array self.AllCharWindowIndexes[w] stores all character 
                # windows within the word w.
                allCharWindows = self.AllCharWindowIndexes[w]
                
                lenWord = len(allCharWindows)
                
                if lenWord >= numMaxCh:
                    # Get only the numMaxCh-character long suffix of the word.
                    charRepr = allCharWindows[lenWord - numMaxCh:]
                else:
                    # Get the whole word and append artificial characters to
                    # fill it up to numMaxCh characters.
                    charRepr = allCharWindows + [artCharWindow] * (numMaxCh - lenWord)
                
                charsOfExample.append(charRepr)

            chars.append(charsOfExample)
        
        return np.asarray(chars, dtype="int32")

    def getAllWordCharWindowIndexesBySentence(self, inputData):
        
        numChar = []            
        charWindowOfWord = []
        maxPosByWord = []
           
        for idxSentence in range(len(inputData)):
            numCharSentence = []
            jj = 0   
            for idxWord in range(len(inputData[idxSentence])):
                WindowIndexes = self.getWindowIndexes(idxWord, inputData[idxSentence])
                
                
                for j in WindowIndexes:
                    for item in self.AllCharWindowIndexes[j]:
                        charWindowOfWord.append(item)
                    numCharSentence.append(self.numCharsOfWord[j])
                    
                    
                    line = []
                    for ii in range(min(self.numCharsOfWord[j], self.maxLenWord)):
                        line.append(jj)
                        jj += 1
                    while ii + 1 < self.maxLenWord:
                        line.append(jj - 1)
                        ii += 1
                    
      
                    maxPosByWord.append(line)     
                    
            numChar.append(numCharSentence)      
        
        
        return [np.array(charWindowOfWord), np.asarray(maxPosByWord, dtype="int64"), np.array(numChar)]

    # esta funcao monta a janela de chars de uma palavra    
    def getAllCharIndexes(self, charIdxWord):
        allWindowIndexes = []
        
        for Idx in xrange(len(charIdxWord)):
            indexes = []
           
            for charIdx in xrange(len(charIdxWord[Idx])):
                indexes.append(self.getCharIndexes(charIdx, charIdxWord[Idx]))
            allWindowIndexes.append(indexes)
            
        return allWindowIndexes;

    def updateAllCharIndexes(self, charIdxWord):
        
        for Idx in xrange(len(charIdxWord)):
            indexes = []
           
            for charIdx in xrange(len(charIdxWord[Idx])):
                indexes.append(self.getCharIndexes(charIdx, charIdxWord[Idx]))
            
            self.AllCharWindowIndexes.append(indexes)

    def getCharIndexes(self, idxWord, data):
        lenData = len(data)
        windowNums = []
        contextSize = int(np.floor((self.charWindowSize - 1) / 2))
        i = idxWord - contextSize
        while (i <= idxWord + contextSize):
            if(i < 0):
                windowNums.append(self.startSymbol)
            elif (i >= lenData):
                windowNums.append(self.endSymbol)
            else:
                windowNums.append(data[i])
            i += 1
        return windowNums
    
    def getWindowIndexes(self, idxWord, data):
        lenData = len(data)
        windowNums = []
        contextSize = int(np.floor((self.wordWindowSize - 1) / 2))
        i = idxWord - contextSize
        while (i <= idxWord + contextSize):
            if(i < 0):
                windowNums.append(self.startSymbol)
            elif (i >= lenData):
                windowNums.append(self.endSymbol)
            else:
                windowNums.append(data[i])
            i += 1
        return windowNums
    
    def confBatchSize(self, inputData, batch):
        
        # Configura o batch size
        
        numWords = len(inputData)
        batchStep = []
        charIdx = 0
        
       
        if self.separateSentence:
            for sentence in inputData:
                batchStep.append(sum(sentence))
    
        else:                
            for idx in range(len(batch)):
                if charIdx >= numWords:
                    break;
                
                step = sum(inputData[charIdx:charIdx + (self.wordWindowSize * batch[idx])])
                
                batchStep.append(step)
                charIdx += (self.wordWindowSize * batch[idx])
            
        return np.asarray(batchStep);

    def getParameters(self):
        return self.__embedLayer.getParameters() + \
            self.hiddenLayer.getParameters()
    
    def getDefaultGradParameters(self):
        return self.hiddenLayer.getDefaultGradParameters() + \
            self.__embedLayer.getDefaultGradParameters()
    
    def getStructuredParameters(self):
        return self.hiddenLayer.getStructuredParameters() + \
            self.__embedLayer.getStructuredParameters()
    
    def getUpdates(self, cost, lr, sumSqGrads=None):
        return self.__embedLayer.getUpdates(cost, lr, sumSqGrads)
    
    def getNormalizationUpdates(self, strategy, coef):
        return self.embedding.getNormalizationUpdate(strategy, coef)

    def getOutput(self):
        return self.output
