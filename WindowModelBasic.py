#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from NNet.HiddenLayer import HiddenLayer
from NNet.WortToVectorLayer import WordToVectorLayer
import math
from NNet.Util import LearningRateUpdNormalStrategy
import time
from numpy import arange
import random

class WindowModelBasic:
    startSymbolStr = "<s>"
    endSymbolStr = "</s>"
    
    @staticmethod
    def setStartSymbol(startSymbol):
        WindowModelBasic.startSymbolStr = startSymbol
        
    @staticmethod
    def setEndSymbol(endSymbol):
        WindowModelBasic.endSymbolStr = endSymbol

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1.0, c=0.0
                 ,learningRateUpdStrategy = LearningRateUpdNormalStrategy(),randomizeInput=False,withoutHiddenLayer = False):
        self.Wv = theano.shared(name='wordVecs',
                                value=np.asarray(wordVectors.getWordVectors(), dtype=theano.config.floatX),
                                borrow=True)
        self.wordSize = wordVectors.getLenWordVector()
        self.lrValue = _lr
        self.lr =  T.dscalar('lr')
        self.learningRateUpdStrategy = learningRateUpdStrategy;
        self.hiddenSize = hiddenSize;
        self.windowSize = windowSize
        self.regularizationFactor = c;
        self.startSymbol = lexicon.getLexiconIndex(WindowModelBasic.startSymbolStr)
        self.endSymbol = lexicon.getLexiconIndex(WindowModelBasic.endSymbolStr)
        self.numClasses = numClasses
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.cost = None
        self.update = None
        self.regularizationFactor = theano.shared(c)
        self.y = theano.shared(np.asarray([0]),"y",borrow=True)
        self.isToRandomizeInput = randomizeInput
        
        # Nós casos em que é feita a predição a cada certo número de épocas de um treinamento,
        # ocorre uma concorrência no uso do atributo windowIdxs pelos métodos predict e train. O problema
        # é que o método predict sobrescreve o valor do windowIdxs, que contém os valores da janela do train.
        # Assim, o atributo reloadWindowIds é verdadeiro toda vez que o método predict é chamado.
        # Se o método train estiver sendo executado e se  o reloadWindowIds for verdadeiro, então o valor do windowsIdx
        # é configurado com os valores da janelas do treinamento. Esta solução só foi pensadam para rodar em ambientes não paralelos
        self.reloadWindowIds = False
        
        self.initWithBasicLayers(withoutHiddenLayer)
        self.listeners = []
        
    def addListener(self,listener):
        self.listeners.append(listener)
        
    def setCost(self,cost):
        self.cost = cost
    
    def setUpdates(self,updates):
        self.updates = updates
    
    def initWithBasicLayers(self,withoutHiddenLayer):
        # Camada: word window.
        self.windowIdxs = theano.shared(value=np.zeros((1,self.windowSize),dtype="int64"),
                                   name="windowIdxs")
        
        # Camada: lookup table.
        self.wordToVector = WordToVectorLayer(self.windowIdxs, self.Wv, self.wordSize, True)
        
        # Camada: hidden layer com a função Tanh como função de ativaçãos
        if not withoutHiddenLayer:
            print 'With Hidden Layer'
            self.hiddenLayer = HiddenLayer(self.wordToVector.getOutput(), self.wordSize * self.windowSize , self.hiddenSize);
        else:
            print 'Without Hidden Layer'
    
    def getAllWindowIndexes(self, data):
        raise NotImplementedError();
    
    def reshapeCorrectData(self,correctData):
        raise NotImplementedError();

    def getWindowIndexes(self, idxWord, data):
        lenData = len(data)
        windowNums = []
        contextSize = int(np.floor((self.windowSize - 1) / 2))
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
    
    def confBatchSize(self,inputData):
        raise NotImplementedError();
     

    def train(self, inputData, correctData):
        self.reloadWindowIds = False
        
        # Label
        self.y.set_value(self.reshapeCorrectData(correctData),borrow=True)
        
        # Camada: word window.
        windowIdxs = self.getAllWindowIndexes(inputData)
        
        self.windowIdxs.set_value(windowIdxs,borrow=True)
        
        batchesSize = self.confBatchSize(inputData)
        
        batchSize = theano.shared(0, "batchSize"); 
        index = T.iscalar("index")
        
        # Train function.
        train = theano.function(inputs=[index,self.lr],
                                outputs=self.cost,
                                updates=self.updates,
                                givens={
                                        self.windowIdxs: self.windowIdxs[index : index + batchSize],
                                        self.y: self.y[index : index + batchSize]
                                })
        
        
        self.beginBlock = []    
        pos = 0
        
        for v in batchesSize:
            self.beginBlock.append(pos)
            pos += v
        
        idxList = range(len(batchesSize))
        
        for ite in range(1,self.numEpochs + 1):
            print 'Epoch ' + str(ite)
            
            if self.isToRandomizeInput:
                random.shuffle(idxList)
            
            lr = self.learningRateUpdStrategy.getCurrentLearninRate(self.lrValue,ite)
            
            t1 = time.time()
        
            for idx in idxList:
                batchSize.set_value(batchesSize[idx])
                train(self.beginBlock[idx],lr) 
                
            print 'Time to training the epoch  ' + str(time.time() - t1)
            
            for l in self.listeners:
                l.afterEpoch(ite)
                
            if self.reloadWindowIds:
                self.windowIdxs.set_value(windowIdxs,borrow=True)
                self.reloadWindowIds = False
            
    def predict(self, inputData):
        raise NotImplementedError();
    
