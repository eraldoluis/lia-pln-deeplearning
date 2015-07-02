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
from CharWNN import *

class WindowModelByWord(WindowModelBasic):

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1, c=0.0,charModel=None):
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize, hiddenSize, _lr, numClasses, numEpochs, batchSize, c,charModel)
        
        if charModel == None:
            # Camada: softmax
            self.softmax = SoftmaxLayer(self.hiddenLayer.getOutput(), self.hiddenSize, numClasses);
        
            # Pega o resultado do foward
            foward = self.softmax.getOutput();
        
            parameters = self.softmax.getParameters() + self.hiddenLayer.getParameters()
            # Custo
            cost = negative_log_likelihood(foward, self.y) + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
            updates = self.hiddenLayer.getUpdate(cost, self.lr);
        
        else:
            # Camada: softmax
            self.softmax = SoftmaxLayer(self.second_hiddenLayer.getOutput(), numClasses, numClasses);
        
            # Pega o resultado do foward
            foward = self.softmax.getOutput();
        
            
            parameters = self.softmax.getParameters() + self.first_hiddenLayer.getParameters()
            parameters += self.second_hiddenLayer.getParameters()
            parameters += self.charModel.hiddenLayer.getParameters()
            
            # Custo 
            cost = negative_log_likelihood(foward, self.y) + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
            self.charModel.setCost(cost)
            self.charModel.setUpdates()
            
            updates = self.first_hiddenLayer.getUpdate(cost, self.lr);
            updates += self.second_hiddenLayer.getUpdate(cost, self.lr);
            updates += self.charModel.updates
            
        
        updates += self.softmax.getUpdate(cost, self.lr);
        updates += self.wordToVector.getUpdate(cost, self.lr); 
        
        self.setCost(cost)
        self.setUpdates(updates)
    
    #Esta funcao retorna todos os indices das janelas de palavras  
    def getAllWindowIndexes(self, data):
        allWindowIndexes = [];
        
        for idxWord in range(len(data)):
            allWindowIndexes.append(self.getWindowIndexes(idxWord, data))
            
        return np.array(allWindowIndexes);

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
        # Configura o batch size
        if isinstance(self.batchSize, list):
            return np.asarray(self.batchSize);
        
        return np.full(numWordsInTrain/self.batchSize + 1,self.batchSize)
        
    def predict(self, inputData):
        
        predict = 0
        if self.charModel == None:
            self.windowIdxs.set_value(self.getAllWindowIndexes(inputData),borrow=True)
            y_pred = self.softmax.getPrediction();
            f = theano.function([],[y_pred]);
            predict = f()[0];
            
        else:
            
            self.charModel.charWindowIdxs.set_value(self.charModel.getAllWordCharWindowIndexes(inputData),borrow=True)
            self.windowIdxs.set_value(self.charModel.allWindowIndexes,borrow=True)
            index = T.iscalar("index")
            f = theano.function(inputs=[index],
                                    outputs=self.softmax.getPrediction(),
                                    givens={
                                            self.charModel.charWindowIdxs: self.charModel.charWindowIdxs[T.sum(self.charModel.numCharByWord[0:index]):T.sum(self.charModel.numCharByWord[0:index+self.windowSize])],
                                            self.charModel.posMaxByWord:self.charModel.posMaxByWord[index*self.windowSize:(index+1)*self.windowSize],
                                            self.windowIdxs: self.windowIdxs[index : index + 1],
                                            
                                    })
            predict = []
            for i in range(len(inputData)):
                predict.append(f(i)[0]);
            
        return predict
        
                
        
