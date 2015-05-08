#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NNet.SentenceSoftmaxLayer import SentenceSoftmaxLayer
from NNet.Util import regularizationSquareSumParamaters
from WindowModelBasic import WindowModelBasic
import numpy as np
from itertools import chain


class WindowModelBySentence(WindowModelBasic):

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1, c=0.0,):
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize, hiddenSize, _lr, numClasses, numEpochs, batchSize, c)
    
        # Camada: softmax
        self.sentenceSoftmax = SentenceSoftmaxLayer(self.hiddenLayer.getOutput(), self.hiddenSize, numClasses);
        
        # Custo
        parameters = self.sentenceSoftmax.getParameters() + self.hiddenLayer.getParameters()
        
        numWords = self.windowIdxs.shape[0]
        
        logOfSumAllPath = self.sentenceSoftmax.getLogOfSumAllPathY(numWords)
        
        negativeLogLikehood = -(self.sentenceSoftmax.getSumPathY(self.y) - logOfSumAllPath)
        cost =   negativeLogLikehood + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
                    
        # Gradiente dos pesos e do bias
        updates = self.hiddenLayer.getUpdate(cost, self.lr);
        updates += self.sentenceSoftmax.getUpdate(cost, self.lr);
        updates += self.wordToVector.getUpdate(cost, self.lr); 
        
        
        self.setCost(cost)
        self.setUpdates(updates)
        
    def reshapeCorrectData(self,correctData):
        return np.fromiter(chain.from_iterable(correctData),dtype=int)
    
    def getAllWindowIndexes(self, data):
        allWindowIndexes = [];
        self.setencesSize = [];
        
        for idxSentence in range(len(data)):
            for idxWord in data[idxSentence]:
                allWindowIndexes.append(self.getWindowIndexes(idxWord, data[idxSentence]))
            
            self.setencesSize.append(len(data[idxSentence]))
        
        return np.array(allWindowIndexes);
    
    def confBatchSize(self,numWordsInTrain):
        # Configura o batch size
        return np.asarray(self.setencesSize)
        
    def predict(self, inputData):
        sentences = self.getAllWindowIndexes(inputData);
        
        predicts = []
               
        for windowSetence in sentences:
            self.windowIdxs.set_value(windowSetence,borrow=True)
            
            predicts.append(self.sentenceSoftmax.predict(len(windowSetence)))
    
        
        return np.asarray(predicts);
