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
from WindowModelBasic import WindowModelBasic

class WindowModelByWord(WindowModelBasic):

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1, c=0.0,):
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize, hiddenSize, _lr, numClasses, numEpochs, batchSize, c)
                
        # Camada: softmax
        self.softmax = SoftmaxLayer(self.hiddenLayer.getOutput(), self.hiddenSize, numClasses);
        
        # Pega o resultado do foward
        foward = self.softmax.getOutput();
        
        # Custo
        parameters = self.softmax.getParameters() + self.hiddenLayer.getParameters()
        cost = negative_log_likelihood(foward, self.y) + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
        
        # Gradiente dos pesos e do bias
        updates = self.hiddenLayer.getUpdate(cost, self.lr);
        updates += self.softmax.getUpdate(cost, self.lr);
        updates += self.wordToVector.getUpdate(cost, self.lr); 
        
        self.setCost(cost)
        self.setUpdates(updates)
    
    def reshapeCorrectData(self,correctData):
        return np.asarray(correctData)
    
    def getAllWindowIndexes(self, data):
        allWindowIndexes = [];
        
        for idxWord in range(len(data)):
            allWindowIndexes.append(self.getWindowIndexes(idxWord, data))
            
        return np.array(allWindowIndexes);
    
    def confBatchSize(self,numWordsInTrain):
        # Configura o batch size
        if isinstance(self.batchSize, list):
            return np.asarray(self.batchSize);
        
        
        
        return np.full(numWordsInTrain/self.batchSize + 1,self.batchSize)
        
    def predict(self, inputData):
        self.windowIdxs.set_value(self.getAllWindowIndexes(inputData),borrow=True)
        
        y_pred = self.softmax.getPrediction();
          
        f = theano.function([],[y_pred]);
        
        return f();
    
