#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from NNet.HiddenLayer import HiddenLayer
from NNet.WortToVectorLayer import WordToVectorLayer
from NNet.Util import LearningRateUpdNormalStrategy
from WindowModelBasic import WindowModelBasic



class CharWNN():

    def __init__(self,charcon, charVectors, charIdxWord, numCharsOfWord, charWindowSize, wordWindowSize , convSize,
                  numClasses, c=0.0,learningRateUpdStrategy = LearningRateUpdNormalStrategy(),separateSentence=False):
        
        
        self.CharIdxWord = charIdxWord
        self.numCharsOfWord = numCharsOfWord
        
        self.Cv = theano.shared(name='charVecs',
                                value=np.asarray(charVectors.getWordVectors(), dtype=theano.config.floatX),
                                borrow=True)
        
        self.charSize = charVectors.getLenWordVector()
        self.lr =  T.dscalar('lr')
        self.charWindowSize = charWindowSize
        self.wordWindowSize = wordWindowSize
        self.startSymbol = charcon.getLexiconIndex(WindowModelBasic.startSymbolStr)
        self.endSymbol = charcon.getLexiconIndex(WindowModelBasic.endSymbolStr)
        self.numClasses = numClasses
        self.separateSentence = separateSentence
        
        self.batchSize = theano.shared(name='charBatchSize', value=1)
        self.updates = None
        self.cost = None
        self.output = None
        #self.allWindowIndexes = None
        self.regularizationFactor = theano.shared(c)
        self.convSize = convSize
        
        self.maxLenWord = max(numCharsOfWord)
        self.posMaxByWord = theano.shared(np.zeros((2,self.maxLenWord),dtype="int64"),'posMaxByWord',int)
        
        # Inicializando as camadas básicas 
        self.initWithBasicLayers()
        
        #Montar uma lista com as janelas de char de cada palavra do dicionario
        self.AllCharWindowIndexes = self.getAllCharIndexes(charIdxWord)
        
        
        #definir a saída da camada Charwnn, a representação da janela de palavras à nível de caracter
        self.output = T.max(self.hiddenLayer.getOutput()[self.posMaxByWord], axis=1).reshape((self.batchSize,self.wordWindowSize * self.convSize))
        
    
    def initWithBasicLayers(self):
        
        # Camada: char window
        self.charWindowIdxs = theano.shared(value=np.zeros((2,self.charWindowSize),dtype="int64"),
                                   name="charWindowIdxs")
                
        # Camada: lookup table.
        self.wordToVector = WordToVectorLayer(self.charWindowIdxs,self.Cv, self.charSize, True)
        
        # Camada: hidden layer com a função Tanh como função de ativaçãos
        #self.hiddenLayer = HiddenLayer(self.wordToVector.getOutput(),self.charSize * self.charWindowSize , self.convSize, W=None, b=None, activation=None);
        self.hiddenLayer = HiddenLayer(self.wordToVector.getOutput(),self.charSize * self.charWindowSize , self.convSize);

    # Esta função retorna o índice das janelas dos caracteres de todas as palavras 
    def getAllWordCharWindowIndexes(self,inputData):
        
        if self.separateSentence:
            return self.getAllWordCharWindowIndexesBySentence(inputData)
        
        return self.getAllWordCharWindowIndexesByWord(inputData)
        
        
    def getAllWordCharWindowIndexesByWord(self,inputData):
        
        numChar = []
        charWindowOfWord = []
        maxPosByWord = []
                
        for idxWord in range(len(inputData)):
            
            WindowIndexes = self.getWindowIndexes(idxWord, inputData)
            
            jj = 0
            
            for j in WindowIndexes:
                for item in self.AllCharWindowIndexes[j]:
                    charWindowOfWord.append(item)
                numChar.append(self.numCharsOfWord[j])
                
                line = []
                for ii in range(self.numCharsOfWord[j]):
                    line.append(jj)
                    jj += 1
                while ii+1 < self.maxLenWord:
                    line.append(jj-1)
                    ii += 1
                maxPosByWord.append(line)     
                       
        
        return [np.array(charWindowOfWord),np.asarray(maxPosByWord),np.array(numChar)]
        
    
    def getAllWordCharWindowIndexesBySentence(self,inputData):
        
        numChar = []            
        charWindowOfWord = []
        maxPosByWord = []
           
        for idxSentence in range(len(inputData)):
            numCharSentence = []   
            for idxWord in range(len(inputData[idxSentence])):
                WindowIndexes = self.getWindowIndexes(idxWord, inputData[idxSentence])
                jj = 0
                
                for j in WindowIndexes:
                    for item in self.AllCharWindowIndexes[j]:
                        charWindowOfWord.append(item)
                    numCharSentence.append(self.numCharsOfWord[j])
                    
                    line = []
                    for ii in range(self.numCharsOfWord[j]):
                        line.append(jj)
                        jj += 1
                    while ii+1 < self.maxLenWord:
                        line.append(jj-1)
                        ii += 1
                    maxPosByWord.append(line)     
                    
            numChar.append(numCharSentence)      
            
        for pp in maxPosByWord:
	  if len(pp)!= self.maxLenWord:
	    print len(pp)
        b = np.asarray(maxPosByWord)
        print b
        print b.shape,posMaxByWord.dtype
        return [np.array(charWindowOfWord),np.matrix(maxPosByWord),np.array(numChar)]

        
    #esta funcao monta a janela de chars de uma palavra    
    def getAllCharIndexes(self,charIdxWord):
        allWindowIndexes = []
        
        for Idx in range(len(charIdxWord)):
            indexes = []
           
            for charIdx in range(len(charIdxWord[Idx])):
                indexes.append(self.getCharIndexes(charIdx,charIdxWord[Idx]))
            allWindowIndexes.append(indexes)
            
        return allWindowIndexes;
    
    def updateAllCharIndexes(self,charIdxWord):
        
        for Idx in range(len(charIdxWord)):
            indexes = []
           
            for charIdx in range(len(charIdxWord[Idx])):
                indexes.append(self.getCharIndexes(charIdx,charIdxWord[Idx]))
                
            self.AllCharWindowIndexes.append(indexes)
    

    def getCharIndexes(self, idxWord,data):
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
    
    def confBatchSize(self,inputData,batch):
        
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
                
                step = sum(inputData[charIdx:charIdx+(self.wordWindowSize*batch[idx])])
                
                batchStep.append(step)
                charIdx += (self.wordWindowSize*batch[idx])
            
        return np.asarray(batchStep);
            #batcheSize==windowSize??
            #for idx in range(len(batch)):
            #    if charIdx >= numWords:
            #        break;
            #    
            #    step = sum(inputData[charIdx:charIdx+batch[idx]])
            #    batchStep.append(step)
            #    charIdx += batch[idx]        
    
    
     
    def setUpdates(self):
        updates = self.hiddenLayer.getUpdate(self.cost, self.lr);    
        updates += self.wordToVector.getUpdate(self.cost, self.lr);
        self.updates = updates
        
    def setCost(self,cost):
        self.cost = cost
        
             
    def getOutput(self):
        return self.output;

