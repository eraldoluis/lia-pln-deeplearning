#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from NNet.HiddenLayer import HiddenLayer
from NNet.EmbeddingLayer import EmbeddingLayer
from NNet.Util import LearningRateUpdNormalStrategy
import time
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

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr, numClasses, numEpochs, batchSize=1.0, c=0.0, charModel=None
                 , learningRateUpdStrategy=LearningRateUpdNormalStrategy(), randomizeInput=False, wordVecsUpdStrategy='normal', withoutHiddenLayer=False, networkAct='tanh', norm_coef=1.0):


        self.Wv = theano.shared(np.asarray(wordVectors.getWordVectors(), dtype=theano.config.floatX),
                                'wordVecs',
                                borrow=True)
        self.wordSize = wordVectors.getLenWordVector()
        self.lrValue = _lr
        self.lr = T.dscalar('lr')
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
        self.y = theano.shared(np.asarray([0], dtype="int32"), "y", borrow=True)
        
        self.wordVecsUpdStrategy = wordVecsUpdStrategy
        self.charModel = charModel
        self.isToRandomizeInput = randomizeInput
        self.numCharByWord = theano.shared(np.asarray([0]), 'numCharByWord', int)
        
        self.networkAct = networkAct
        self.norm_coef = norm_coef
        # Nos casos em que é feita a predição a cada certo número de épocas de um treinamento,
        # ocorre uma concorrência no uso do atributo windowIdxs pelos métodos predict e train. O problema
        # é que o método predict sobrescreve o valor do windowIdxs, que contém os valores da janela do train.
        # Assim, o atributo reloadWindowIds é verdadeiro toda vez que o método predict é chamado.
        # Se o método train estiver sendo executado e se  o reloadWindowIds for verdadeiro, então o valor do windowsIdx
        # é configurado com os valores da janelas do treinamento. Esta solução só foi pensadam para rodar em ambientes não paralelos
        self.reloadWindowIds = False
    
        self.initWithBasicLayers(withoutHiddenLayer)

        self.listeners = []
        
    def addListener(self, listener):
        self.listeners.append(listener)
        
    def setCost(self, cost):
        self.cost = cost
    
    def setUpdates(self, updates):
        self.updates = updates
    
    def initWithBasicLayers(self, withoutHiddenLayer):
        # Variável que representa uma entrada para a rede. A primeira camada é 
        # a camada de embedding que recebe um batch de janelas de palavras.
        # self.windowIdxs = theano.shared(value=np.zeros((1, self.windowSize), dtype="int64"), name="windowIdxs")
        self.windowIdxs = T.imatrix("windowIdxs")

        # Word embedding layer
        # self.embedding = EmbeddingLayer(self.windowIdxs, self.Wv, self.wordSize, True,self.wordVecsUpdStrategy,self.norm_coef)
        self.embedding = EmbeddingLayer(self.windowIdxs, self.Wv)
        
        # Camada: hidden layer com a função Tanh como função de ativaçãos
        if not withoutHiddenLayer:
            print 'With Hidden Layer'
            
            if self.charModel == None:
                # Camada: hidden layer com a função Tanh como função de ativação
                self.hiddenLayer = HiddenLayer(self.embedding.getOutput(),
                                               self.wordSize * self.windowSize,
                                               self.hiddenSize,
                                               activation=self.networkAct)
            else:    
                # Camada: hidden layer com a função Tanh como função de ativação
                self.hiddenLayer = HiddenLayer(T.concatenate([self.embedding.getOutput(),
                                                              self.charModel.getOutput()],
                                                             axis=1),
                                               (self.wordSize + self.charModel.convSize) * self.windowSize,
                                               self.hiddenSize,
                                               activation=self.networkAct);
            
        else:
            print 'Without Hidden Layer'
    
    def getAllWindowIndexes(self, data):
        raise NotImplementedError();
    
    def reshapeCorrectData(self, correctData):
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
    
    def confBatchSize(self, inputData):
        raise NotImplementedError();
    
    def train(self, inputData, correctData, indexesOfRawWord):
        self.reloadWindowIds = False
        
        charIndex = T.iscalar("charIndex")
        charWindowIdxs = None
        posMaxByWord = None
        numCharByWord = None
        
        self.charBeginBlock = None

        # Labels
        self.y.set_value(self.reshapeCorrectData(correctData), borrow=True)

        # Matrix with training data.
        windowIdxs = theano.shared(self.getAllWindowIndexes(inputData), borrow=True)

        batchesSize = self.confBatchSize(inputData)

        # This is the index of the batch to be trained.
        # It is multiplied by self.batchSize to provide the correct slice
        # for each training iteration.
        batchIndex = T.iscalar("batchIndex")

        charBatchesSize = None
        charBatchS = theano.shared(1, "charBatchS")

        # Train function.
        if self.charModel == None:
            train = theano.function(inputs=[batchIndex, self.lr],
                                    outputs=self.cost,
                                    updates=self.updates,
                                    givens={     
                                            self.windowIdxs: windowIdxs[batchIndex * self.batchSize : (batchIndex + 1) * self.batchSize],
                                            self.y: self.y[batchIndex * self.batchSize : (batchIndex + 1) * self.batchSize]
                                    })  # , mode='DebugMode')
        else:
            charmodelIdxsPos = self.charModel.getAllWordCharWindowIndexes(indexesOfRawWord)
            charWindowIdxs = charmodelIdxsPos[0]
            posMaxByWord = charmodelIdxsPos[1]
            numCharByWord = charmodelIdxsPos[2]

            self.charModel.charWindowIdxs.set_value(charWindowIdxs, borrow=True)
            self.charModel.posMaxByWord.set_value(posMaxByWord, borrow=True)
            
            charBatchesSize = self.charModel.confBatchSize(numCharByWord, batchesSize)
            
                    
            train = theano.function(inputs=[batchIndex, self.lr, self.charModel.lr, charIndex],
                                    outputs=self.cost,
                                    updates=self.updates,
                                    givens={
                                            self.charModel.charWindowIdxs: self.charModel.charWindowIdxs[charIndex:charIndex + charBatchS],
                                            self.charModel.posMaxByWord:self.charModel.posMaxByWord[batchIndex * self.windowSize:(batchIndex + batchSize) * self.windowSize],
                                            self.windowIdxs: self.windowIdxs[batchIndex : batchIndex + batchSize],
                                            self.y: self.y[batchIndex : batchIndex + batchSize]
                                    })
            
            self.charBeginBlock = []    
            pos = 0
            
            for v in charBatchesSize:
                self.charBeginBlock.append(pos)
                pos += v
                
                    
        self.beginBlock = []    
        pos = 0
        
        for v in batchesSize:
            self.beginBlock.append(pos)
            pos += v
        
                
        idxList = range(len(batchesSize))
        
        
        
        for ite in range(1, self.numEpochs + 1):
            print 'Epoch ' + str(ite)
            
            if self.isToRandomizeInput:
                random.shuffle(idxList)
            
            lr = self.learningRateUpdStrategy.getCurrentLearninRate(self.lrValue, ite)
            
            t1 = time.time()

            if self.charModel == None:                
                for idx in idxList:
                    # batchSize.set_value(batchesSize[idx])
                    # train(self.beginBlock[idx], lr)     
                    train(idx, lr)
            else:
                for idx in idxList:
                    # batchSize.set_value(batchesSize[idx])
                    charBatchS.set_value(charBatchesSize[idx])
     
                    self.charModel.batchSize.set_value(batchesSize[idx])

                    train(self.beginBlock[idx], lr, lr, self.charBeginBlock[idx]) 

            print 'Time to training the epoch  ' + str(time.time() - t1)
            
            # Evaluate the model if the iteration is in the list of epochs to evaluate
            for l in self.listeners:
                l.afterEpoch(ite)
                
            if self.reloadWindowIds:
                self.windowIdxs.set_value(windowIdxs, borrow=True)
                
                if self.charModel is not None:
                    self.charModel.charWindowIdxs.set_value(charWindowIdxs, borrow=True)
                    self.charModel.posMaxByWord.set_value(posMaxByWord, borrow=True)
                    
                self.reloadWindowIds = False
                
    def predict(self, inputData):
        raise NotImplementedError();
    

