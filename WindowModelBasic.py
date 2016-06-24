#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.compile
import theano.tensor as T
from NNet.HiddenLayer import HiddenLayer
from NNet.EmbeddingLayer import EmbeddingLayer
from NNet.Util import LearningRateUpdNormalStrategy
import time
import random
import logging

class WindowModelBasic:
    startSymbolStr = "<s>"
    endSymbolStr = "</s>"
    
    @staticmethod
    def setStartSymbol(startSymbol):
        WindowModelBasic.startSymbolStr = startSymbol

    @staticmethod
    def setEndSymbol(endSymbol):
        WindowModelBasic.endSymbolStr = endSymbol
        
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_WindowModelBasic__log']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
        self.__log = logging.getLogger(__name__)

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,
                 numClasses, numEpochs, batchSize=1.0, c=0.0, charModel=None,
                 learningRateUpdStrategy=LearningRateUpdNormalStrategy(),
                 randomizeInput=False, wordVecsUpdStrategy='normal',
                 withoutHiddenLayer=False, networkAct='tanh', norm_coef=1.0,
                 structGrad=True, adaGrad=False):
        # Logging object.
        self.__log = logging.getLogger(__name__)

        self.Wv = []
        
        if not isinstance(wordVectors, list):
            wordVectors = [wordVectors]
        
        a = 1
        self.wordSize = 0
        
        for wv in wordVectors:
            self.Wv.append(theano.shared(name='wordVec' + str(a),
                                value=np.asarray(wv.getWordVectors(), dtype=theano.config.floatX),
                                borrow=True))
            
            self.wordSize += wv.getLenWordVector()
            
            a+=1
        
        
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
        
        self.wordVecsUpdStrategy = wordVecsUpdStrategy
        self.charModel = charModel
        self.isToRandomizeInput = randomizeInput
        self.numCharByWord = theano.shared(np.asarray([0]), 'numCharByWord', int)
        
        self.networkAct = networkAct
        self.norm_coef = norm_coef
        
        self.__structGrad = structGrad
        self.__adaGrad = adaGrad
        
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
        
        # Variável que representa a saída esperada para cada exemplo do (mini) batch.
        self.y = T.ivector("y")
        
        # Word embedding layer
        # self.embedding = EmbeddingLayer(self.windowIdxs, self.Wv, self.wordSize, True,self.wordVecsUpdStrategy,self.norm_coef)
#         self.embedding = EmbeddingLayer(examples=self.windowIdxs,
#                                         embedding=self.Wv,
#                                         structGrad=self.__structGrad)
        
        # Camada: lookup table.
        self.embeddings = []
        
        for Wv in self.Wv:
            embedding = EmbeddingLayer(examples=self.windowIdxs,
                                embedding=Wv,
                                structGrad=self.__structGrad)
            self.embeddings.append(embedding)
        
        concatenateEmbeddings = T.concatenate( [embeddingLayer.getOutput() for embeddingLayer in self.embeddings],axis=1)
        
        
        # Camada: hidden layer com a função Tanh como função de ativaçãos
        if not withoutHiddenLayer:
            print 'With Hidden Layer'
            
            if self.charModel == None:
                # Camada: hidden layer com a função Tanh como função de ativação
                self.hiddenLayer = HiddenLayer(concatenateEmbeddings,
                                               self.wordSize * self.windowSize,
                                               self.hiddenSize,
                                               activation=self.networkAct)
            else:
                # Camada: hidden layer com a função Tanh como função de ativação
                self.hiddenLayer = HiddenLayer(T.concatenate([concatenateEmbeddings,
                                                              self.charModel.getOutput()],
                                                             axis=1),
                                               (self.wordSize + self.charModel.convSize) * self.windowSize,
                                               self.hiddenSize,
                                               activation=self.networkAct)
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
        # TODO: test
#         n = 1000
#         inputData = inputData[:n]
#         correctData = correctData[:n]
#         indexesOfRawWord = indexesOfRawWord[:n]
        
        log = self.__log

        # Matrix with training data.
        # windowIdxs = theano.shared(self.getAllWindowIndexes(inputData), borrow=True)
        log.info("Generating all training windows (training input)...")
        windowIdxs = theano.shared(self.getAllWindowIndexes(inputData),
                                   name="x_shared",
                                   borrow=True)

        # Correct labels.
        y = theano.shared(self.reshapeCorrectData(correctData),
                          name="y_shared",
                          borrow=True)
        
        numExs = len(inputData)
        if self.batchSize > 0:
            numBatches = numExs / self.batchSize
            if numBatches <= 0:
                numBatches = 1
                self.batchSize = numExs
            elif numExs % self.batchSize > 0:
                numBatches += 1
        else:
            numBatches = 1
            self.batchSize = numExs

        log.info("Training with %d examples using %d mini-batches (batchSize=%d)..." % (numExs, numBatches, self.batchSize))
        
        # This is the index of the batch to be trained.
        # It is multiplied by self.batchSize to provide the correct slice
        # for each training iteration.
        batchIndex = T.iscalar("batchIndex")
        
        # Word-level training data (input and output).
        
        # Character-level training data.
        if self.charModel:
            charInput = theano.shared(self.charModel.getAllWordCharWindowIndexes(indexesOfRawWord), borrow=True)
            givens = {
                      self.windowIdxs: windowIdxs[batchIndex * self.batchSize : (batchIndex + 1) * self.batchSize],
                      self.y: y[batchIndex * self.batchSize : (batchIndex + 1) * self.batchSize],
                      self.charModel.charWindowIdxs : charInput[batchIndex * self.batchSize : (batchIndex + 1) * self.batchSize]
                     }
        else:
            givens = {
                      self.windowIdxs: windowIdxs[batchIndex * self.batchSize : (batchIndex + 1) * self.batchSize],
                      self.y: y[batchIndex * self.batchSize : (batchIndex + 1) * self.batchSize],
                     }
        
        # Train function.
        # train = theano.function(inputs=[self.windowIdxs, batchIndex, self.lr],
        train = theano.function(inputs=[batchIndex, self.lr],
                                outputs=self.cost,
                                updates=self.updates,
                                givens=givens)
        
        # Indexes of all training (mini) batches.
        idxList = range(numBatches)
        
        for epoch in range(1, self.numEpochs + 1):
            print 'Training epoch ' + str(epoch) + '...'
            
            if self.isToRandomizeInput:
                # Shuffle order of mini-batches for this epoch.
                random.shuffle(idxList)
            
            # Compute current learning rate.
            lr = self.learningRateUpdStrategy.getCurrentLearninRate(self.lrValue, epoch)
            
            t1 = time.time()
            
            # Train each mini-batch.
            for idx in idxList:
                train(idx, lr)
                # train(windowIdxs[idx * self.batchSize : (idx + 1) * self.batchSize], idx, lr)    
            
            print 'Time to training the epoch  ' + str(time.time() - t1)
            
            # Evaluate the model when necessary
            for l in self.listeners:
                l.afterEpoch(epoch)
    
    def predict(self, inputData):
        raise NotImplementedError();

    def isAdaGrad(self):
        return self.__adaGrad
    