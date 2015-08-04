#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NNet.SentenceSoftmaxLayer import SentenceSoftmaxLayer
from NNet.Util import regularizationSquareSumParamaters,\
    LearningRateUpdNormalStrategy
from WindowModelBasic import WindowModelBasic
import numpy as np
from itertools import chain

class NeuralNetworkChoiceEnum:
    COMPLETE = 1
    WITHOUT_HIDDEN_LAYER_AND_UPD_WV = 2
    WITHOUT_UPD_WV = 3
    
    @staticmethod
    def withoutHiddenLayer(choice):
        return choice == NeuralNetworkChoiceEnum.WITHOUT_HIDDEN_LAYER_AND_UPD_WV
    
    @staticmethod
    def withoutUpdateWv(choice):
        return choice == NeuralNetworkChoiceEnum.WITHOUT_HIDDEN_LAYER_AND_UPD_WV or choice == NeuralNetworkChoiceEnum.WITHOUT_UPD_WV 


class WindowModelBySentence(WindowModelBasic):

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1, c=0.0
                    ,learningRateUpdStrategy = LearningRateUpdNormalStrategy(), choice = NeuralNetworkChoiceEnum.COMPLETE):
            
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize, hiddenSize, _lr, 
                                  numClasses, numEpochs, batchSize, c,learningRateUpdStrategy,True,NeuralNetworkChoiceEnum.withoutHiddenLayer(choice))
    
        # Camada: softmax
        
        if NeuralNetworkChoiceEnum.withoutHiddenLayer(choice):
            print 'Softmax linked with w2v'
            self.sentenceSoftmax = SentenceSoftmaxLayer(self.wordToVector.getOutput(), self.wordSize * self.windowSize, numClasses);
            parameters = self.sentenceSoftmax.getParameters()
        else:
            print 'Softmax linked with hidden'
            self.sentenceSoftmax = SentenceSoftmaxLayer(self.hiddenLayer.getOutput(), self.hiddenSize, numClasses);
            parameters = self.sentenceSoftmax.getParameters() + self.hiddenLayer.getParameters()
            
        # Custo
        
        logOfSumAllPath = self.sentenceSoftmax.getLogOfSumAllPathY()
        negativeLogLikehood = -(self.sentenceSoftmax.getSumPathY(self.y) - logOfSumAllPath)
        cost =   negativeLogLikehood + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
                      
        # Gradiente dos pesos e do bias
        updates = self.sentenceSoftmax.getUpdate(cost, self.lr);
        
        if not NeuralNetworkChoiceEnum.withoutHiddenLayer(choice):
            print 'With update hidden layer vector'
            updates += self.hiddenLayer.getUpdate(cost, self.lr);
        else:
            print 'Without update hidden layer vector'
            
        if not NeuralNetworkChoiceEnum.withoutUpdateWv(choice):
            print 'With update vector'
            updates += self.wordToVector.getUpdate(cost, self.lr); 
        else:
            print 'Without update vector'
        
        self.setCost(cost)
        self.setUpdates(updates)
        
    def reshapeCorrectData(self,correctData):
        return np.fromiter(chain.from_iterable(correctData),dtype=int)
    
    def getAllWindowIndexes(self, data):
        allWindowIndexes = [];
        self.setencesSize = [];
        
        for idxSentence in range(len(data)):
            for idxWord in range(len(data[idxSentence])):
                allWindowIndexes.append(self.getWindowIndexes(idxWord, data[idxSentence]))
            
            self.setencesSize.append(len(data[idxSentence]))
        
        return np.array(allWindowIndexes);
    
    def confBatchSize(self,inputData):
        # Configura o batch size
        return np.asarray(self.setencesSize)
    
    def predict(self, inputData):
        windowSetences = self.getAllWindowIndexes(inputData);
        
        predicts = []
        index = 0
        indexSentence = 0
        self.reloadWindowIds = True
        
        while index < len(windowSetences):
            self.windowIdxs.set_value(windowSetences[index:index + self.setencesSize[indexSentence]],borrow=True)
            
            predicts.append(self.sentenceSoftmax.predict(self.setencesSize[indexSentence]))
            
            index += self.setencesSize[indexSentence]
            indexSentence += 1
        
        return np.asarray(predicts);
