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

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1.0, c=0.0):
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
        
        # Camada: hidden layer com a função Tanh como função de ativaçãos
        self.hiddenLayer = HiddenLayer(self.wordToVector.getOutput(), self.wordSize * self.windowSize , self.hiddenSize);
        
    
    def getAllWindowIndexes(self, data):
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
        
        # Train function.
        train = theano.function(inputs=[index],
                                outputs=self.cost,
                                updates=self.updates,
                                givens={
                                        self.windowIdxs: self.windowIdxs[index : index + batchSize],
                                        self.y: self.y[index : index + batchSize]
                                })
        
        for ite in range(self.numEpochs):
            minibatch_index = 0
            i = 0
            
            while minibatch_index < numWordsInTrain:
                batchSize.set_value(batchesSize[i])
                
                train(minibatch_index)
                
                minibatch_index += batchesSize[i]
                i+=1
        
    def predict(self, inputData):
        raise NotImplementedError();
    
