#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
from NNet.SoftmaxLayer import SoftmaxLayer
from NNet.Util import negative_log_likelihood, regularizationSquareSumParamaters, \
    LearningRateUpdNormalStrategy
from WindowModelBasic import WindowModelBasic
import theano.tensor as T

class WindowModelByWord(WindowModelBasic):

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr, numClasses, numEpochs, batchSize=1, c=0.0,
                 charModel=None, learningRateUpdStrategy=LearningRateUpdNormalStrategy(), wordVecsUpdStrategy='normal', networkAct='tanh', norm_coef=1.0):
        
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize, hiddenSize, _lr, numClasses, numEpochs,
                                  batchSize, c, charModel, learningRateUpdStrategy, False, wordVecsUpdStrategy, False, networkAct, norm_coef)

        self.setTestValues = True
        
        
        # Camada: softmax
        self.softmax = SoftmaxLayer(self.hiddenLayer.getOutput(), self.hiddenSize, numClasses);
        
        # Pega o resultado do foward
        foward = self.softmax.getOutput();
        
        parameters = self.softmax.getParameters() + self.hiddenLayer.getParameters()
        
        
        
        if charModel == None:
            
            # Custo
            cost = negative_log_likelihood(foward, self.y) + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
            updates = self.hiddenLayer.getUpdate(cost, self.lr);
        
        else:
            
            parameters += self.charModel.hiddenLayer.getParameters()
            
            # Custo 
            cost = negative_log_likelihood(foward, self.y) + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
            self.charModel.setCost(cost)
            self.charModel.setUpdates()
            
            updates = self.hiddenLayer.getUpdate(cost, self.lr);
            updates += self.charModel.updates

        updates += self.softmax.getUpdate(cost, self.lr)
        updates += self.wordToVector.getUpdate(cost, self.lr)
        
        # Add normalization updates.
        if (self.wordVecsUpdStrategy != 'normal'):
            self.wordToVector.getNormalizationUpdate(self.wordVecsUpdStrategy, self.norm_coef)

        self.setCost(cost)
        self.setUpdates(updates)
    
    
    def reshapeCorrectData(self, correctData):
        return np.asarray(correctData)
      
    # Esta funcao retorna todos os indices das janelas de palavras  
    def getAllWindowIndexes(self, data):
        allWindowIndexes = [];
        
        for idxWord in range(len(data)):
            allWindowIndexes.append(self.getWindowIndexes(idxWord, data))
            
        return np.array(allWindowIndexes);
    
    def confBatchSize(self, inputData):
        numWords = len(inputData)
        
        # Configura o batch size
        if isinstance(self.batchSize, list):
            if sum(self.batchSize) < numWords:
                print "The number of words for training set by batch is smaller than the number of words in inputData"
            else:
                raise Exception("The total number of words in batch exceeds the number of words in inputData")
            
            return np.asarray(self.batchSize);

        num = numWords / self.batchSize  
        arr = np.full(num, self.batchSize, dtype=np.int64)
        if numWords % self.batchSize:
            arr = np.append(arr, numWords % self.batchSize)
        
        return arr
        # return np.full(numWords/self.batchSize + 1,self.batchSize,dtype=np.int64)
        
    def predict(self, inputData, indexesOfRawWord, unknownDataTest):
      
        self.reloadWindowIds = True  
        predict = 0
        
        if self.setTestValues:
            self.testWordWindowIdxs = self.getAllWindowIndexes(inputData)
            
            if self.charModel:
                self.charModel.updateAllCharIndexes(unknownDataTest)
            
                charmodelIdxPos = self.charModel.getAllWordCharWindowIndexes(indexesOfRawWord)
                
                self.testCharWindowIdxs = charmodelIdxPos[0]
                self.testPosMaxByWord = charmodelIdxPos[1]
                self.testNumCharByWord = charmodelIdxPos[2]
                
            self.setTestValues = False    
        
        self.windowIdxs.set_value(self.testWordWindowIdxs, borrow=True)
        if self.charModel == None:
            
            y_pred = self.softmax.getPrediction();
            f = theano.function([], [y_pred]);
            
            predict = f()[0]
            
        else:

            self.charModel.charWindowIdxs.set_value(self.testCharWindowIdxs, borrow=True)
            self.charModel.posMaxByWord.set_value(self.testPosMaxByWord, borrow=True)
            
            index = T.iscalar("index")
            charIndex = T.iscalar("charIndex")
            step = T.iscalar("step")
            
            
            self.charModel.batchSize.set_value(1)
            
            f = theano.function(inputs=[index, charIndex, step],
                                    outputs=self.softmax.getPrediction(),
                                    givens={
                                            self.charModel.charWindowIdxs: self.charModel.charWindowIdxs[charIndex:charIndex + step],
                                            self.charModel.posMaxByWord:self.charModel.posMaxByWord[index * self.windowSize:(index + 1) * self.windowSize],
                                            self.windowIdxs: self.windowIdxs[index : index + 1],
                                            
                                    })
            predict = []
            j = 0
            for i in range(len(inputData)):
                predict.append(f(i, j, sum(self.testNumCharByWord[i * self.windowSize:(i + 1) * self.windowSize]))[0]);
                j += sum(self.testNumCharByWord[i * self.windowSize:(i + 1) * self.windowSize])
            
            
        return predict
        
