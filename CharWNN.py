#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from NNet.HiddenLayer import HiddenLayer
from NNet.WortToVectorLayer import WordToVectorLayer
#from theano.tensor.nnet.nnet import softmax
#from NNet.SoftmaxLayer import SoftmaxLayer
#from NNet.Util import negative_log_likelihood, regularizationSquareSumParamaters
#from WindowModelBasic import WindowModelBasic
from NNet.Util import defaultGradParameters,WeightTanhGenerator
#from atk import Window


class CharWNN():

    def __init__(self,charIdxWord, numCharsOfWord,charcon, charVectors , charWindowSize, wordWindowSize , convSize, _lr,numClasses,numEpochs,maxLenWord, batchSize=1, c=0.0,):
        
        
        #self.CharIdxWord = charIdxWord
        self.numCharsOfWord = numCharsOfWord
        self.numCharByWord = theano.shared(np.asarray([0]),'numCharByWord',int)
        self.posMaxByWord = theano.shared(np.zeros((1,maxLenWord),dtype="int64"),'posMaxByWord',int)
        self.Cv = theano.shared(name='charVecs',
                                value=np.asarray(charVectors.getWordVectors(), dtype=theano.config.floatX),
                                borrow=True)
        self.charSize = charVectors.getLenWordVector()
        self.lr = _lr
        self.charWindowSize = charWindowSize
        self.wordWindowSize = wordWindowSize
        self.startSymbol = charcon.getLexiconIndex("<s>")
        self.endSymbol = charcon.getLexiconIndex("</s>")
        self.numClasses = numClasses
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.updates = None
        self.cost = None
        self.output = None
        self.allWindowIndexes = None
        self.regularizationFactor = theano.shared(c)
        self.convSize = convSize
        self.weightTanhGenerator = WeightTanhGenerator()
        
        self.maxLenWord = maxLenWord
        
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
        numChar = []
        allWindowIndexes = []
        charWindowOfWord = []
        maxPosByWord = []
                
        
        for idxWord in range(len(inputData)-1):
            allWindowIndexes.append(self.getWindowIndexes(idxWord, inputData))
            for item in self.AllCharWindowIndexes[allWindowIndexes[idxWord][0]]:
                charWindowOfWord.append(item)
            jj = 0    
            for j in allWindowIndexes[idxWord]:
                line = []
                for ii in range(self.numCharsOfWord[j]):
                    line.append(jj)
                    jj += 1
                while ii+1 < self.maxLenWord:
                    line.append(jj-1)
                    ii += 1
                maxPosByWord.append(line)
            numChar.append(self.numCharsOfWord[allWindowIndexes[idxWord][0]])
        jj = 0    
        allWindowIndexes.append(self.getWindowIndexes(idxWord+1, inputData))    
        for j in allWindowIndexes[idxWord+1]:
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
                 
              
        self.allWindowIndexes = np.array(allWindowIndexes)         
        self.numCharByWord.set_value(np.asarray(numChar),borrow=True)
        self.posMaxByWord.set_value(np.asarray(maxPosByWord),borrow=True)
                
            
        return np.array(charWindowOfWord)
    
    #esta funcao monta a janela de chars de uma palavra    
    def getAllCharIndexes(self,charIdxWord):
        allWindowIndexes = []
        
        for Idx in range(len(charIdxWord)):
            indexes = []
           
            for charIdx in range(len(charIdxWord[Idx])):
                indexes.append(self.getCharIndexes(charIdx,charIdxWord[Idx]))
            allWindowIndexes.append(indexes)
            
        return allWindowIndexes;

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
    
    def confBatchSize(self,numWordsInTrain):
        raise NotImplementedError();
     
    def setUpdates(self):
        updates = self.hiddenLayer.getUpdate(self.cost, self.lr);    
        updates += self.wordToVector.getUpdate(self.cost, self.lr);
        self.updates = updates
        
    def setCost(self,cost):
        self.cost = cost
        
    def setLr(self,_lr):
        self.lr = _lr
         
    def getOutput(self):
        return self.output;

