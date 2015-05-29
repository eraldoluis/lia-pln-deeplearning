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

class WindowModelBasic:

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1.0, c=0.0,charModel=None):
        self.Wv = theano.shared(name='wordVecs',
                                value=np.asarray(wordVectors.getWordVectors(), dtype=theano.config.floatX),
                                borrow=True)
        self.wordSize = wordVectors.getLenWordVector()
        self.lr = _lr
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
        if self.charModel:
            self.charModel.setCost(cost)
    
    def setUpdates(self,updates):
        if self.charModel:
            self.charModel.setUpdates()
            updates += self.charModel.updates
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
            #[wordIdx,charEmbedd,charIdx] = self.charModel.getCharEmbeddingOfWindow(self.wordIndex,self.charIndex)
            #self.newCharIndex = charIdx
            # Camada: hidden layer com a função Tanh como função de ativaçãos
            self.first_hiddenLayer = HiddenLayer(T.concatenate([self.wordToVector.getOutput(),self.charModel.getOutput()]), (self.wordSize + self.charModel.convSize) * self.windowSize , self.hiddenSize);
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
        
        # Camada: word window.
        self.windowIdxs.set_value(self.getAllWindowIndexes(inputData),borrow=True)
        
        batchesSize = self.confBatchSize(numWordsInTrain)
        
        batchSize = theano.shared(0, "batchSize"); 
        index = T.iscalar("index")
        charIndex = T.iscalar("charIndex")
        # Train function.
        if self.charModel==None:
            train = theano.function(inputs=[index],
                                    outputs=self.cost,
                                    updates=self.updates,
                                    givens={
                                
                                            self.windowIdxs: self.windowIdxs[index : index + batchSize],
                                            self.y: self.y[index : index + batchSize]
                                    })
        
        else:
            #self.updates += self.charModel.updates
            
            train = theano.function(inputs=[index],
                                    outputs=self.cost,
                                    updates=self.updates,
                                    givens={
                                            #self.wordIndex: index,
                                            #self.charIndex: charIndex,  
                                            self.windowIdxs: self.windowIdxs[index : index + batchSize],
                                            self.y: self.y[index : index + batchSize]
                                    })
    
        for ite in range(self.numEpochs):
            minibatch_index = 0
            i = 0
            #self.newCharIndex = 0
            while minibatch_index < numWordsInTrain:
                batchSize.set_value(batchesSize[i])
                
                train(minibatch_index)
                
                minibatch_index += batchesSize[i]
                i+=1
                
            self.setLr(self.lr/float(ite+1.0))
            if self.charModel:
                self.charModel.setIndexNull()    
        
    def predict(self, inputData):
        raise NotImplementedError();
    
    def setLr(self,__lr):
        self.lr = __lr
        if self.charModel:
            self.charModel.setLr(__lr)
