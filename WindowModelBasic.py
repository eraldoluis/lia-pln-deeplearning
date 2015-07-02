#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from NNet.HiddenLayer import HiddenLayer
from NNet.WortToVectorLayer import WordToVectorLayer
from theano.tensor.nnet.nnet import softmax
from NNet.SoftmaxLayer import SoftmaxLayer
from NNet.Util import negative_log_likelihood, regularizationSquareSumParamaters
import time

class WindowModelBasic:

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1.0, c=0.0,charModel=None):
        self.Wv = theano.shared(name='wordVecs',
                                value=np.asarray(wordVectors.getWordVectors(), dtype=theano.config.floatX),
                                borrow=True)
        self.wordSize = wordVectors.getLenWordVector()
        self.lr = _lr
        self.lr_fixo = _lr
        self.hiddenSize = hiddenSize;
        self.windowSize = windowSize
        self.regularizationFactor = c;
        self.startSymbol = lexicon.getLexiconIndex("<s>")
        self.endSymbol = lexicon.getLexiconIndex("</s>")
        self.numClasses = numClasses
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.cost = None
        self.update = None
        self.regularizationFactor = theano.shared(c)
        self.y = theano.shared(np.asarray([0]),"y",borrow=True)
        self.charModel = charModel
        
        self.initWithBasicLayers()
        
    def setCost(self,cost):
        self.cost = cost
    
    def setUpdates(self,updates):
        self.updates = updates
    
    def initWithBasicLayers(self):
        # Camada: word window.
        self.windowIdxs = theano.shared(value=np.zeros((1,self.windowSize),dtype="int64"),
                                   name="windowIdxs")
        
        
        # Camada: lookup table.
        self.wordToVector = WordToVectorLayer(self.windowIdxs, self.Wv, self.wordSize, True)
        
        if self.charModel == None:
            # Camada: hidden layer com a função Tanh como função de ativaçãos
            self.hiddenLayer = HiddenLayer(self.wordToVector.getOutput(), self.wordSize * self.windowSize , self.hiddenSize);
        else:    
            # Camada: 2 hidden layer com a função Tanh como função de ativaçãos
            self.first_hiddenLayer = HiddenLayer(T.concatenate([self.wordToVector.getOutput(),self.charModel.getOutput()],axis=1), (self.wordSize + self.charModel.convSize) * self.windowSize , self.hiddenSize);
            self.second_hiddenLayer = HiddenLayer(self.first_hiddenLayer.getOutput(), self.hiddenSize , self.numClasses);
        
    
    def getAllWindowIndexes(self, data):
        raise NotImplementedError();

    def getWindowIndexes(self, idxWord, data):
        raise NotImplementedError();
    
    def confBatchSize(self,numWordsInTrain):
        raise NotImplementedError();
     

    def train(self, inputData, correctData):
        numWordsInTrain = len(inputData)
                      
        # Label
        self.y.set_value(np.asarray(correctData),borrow=True)
                       
        batchesSize = self.confBatchSize(numWordsInTrain)
        
        batchSize = theano.shared(1, "batchSize"); 
        index = T.iscalar("index")
        
        # Train function.
        if self.charModel==None:
            # Camada: word window.
            self.windowIdxs.set_value(self.getAllWindowIndexes(inputData),borrow=True)
            train = theano.function(inputs=[index],
                                    outputs=self.cost,
                                    updates=self.updates,
                                    givens={     
                                            self.windowIdxs: self.windowIdxs[index : index + batchSize],
                                            self.y: self.y[index : index + batchSize]
                                    })
        
        else:
            
            
            self.charModel.charWindowIdxs.set_value(self.charModel.getAllWordCharWindowIndexes(inputData),borrow=True)
            self.windowIdxs.set_value(self.charModel.allWindowIndexes,borrow=True)
            train = theano.function(inputs=[index],
                                    outputs=self.cost,
                                    updates=self.updates,
                                    givens={
                                            self.charModel.charWindowIdxs: self.charModel.charWindowIdxs[T.sum(self.charModel.numCharByWord[0:index]):T.sum(self.charModel.numCharByWord[0:index+self.windowSize])],
                                            self.charModel.posMaxByWord:self.charModel.posMaxByWord[index*self.windowSize:(index+1)*self.windowSize], 
                                            self.windowIdxs: self.windowIdxs[index : index + batchSize],
                                            self.y: self.y[index : index + batchSize]
                                    })
    
        for ite in range(self.numEpochs):
            minibatch_wordIndex = 0
            print 'Epoch ' + str(ite+1)
            
            i = 0
            t1 = time.time()
            
        
            while minibatch_wordIndex < numWordsInTrain:
                batchSize.set_value(batchesSize[i])
                
                train(minibatch_wordIndex)
                
                minibatch_wordIndex += batchesSize[i]
                i+=1
                
            self.setLr(self.lr_fixo/float(ite+2.0))
            print 'Time to training the epoch  ' + str(time.time() - t1)
                
        
    def predict(self, inputData):
        raise NotImplementedError();
    
    def setLr(self,__lr):
        self.lr = __lr
        if self.charModel:
            self.charModel.setLr(__lr)
