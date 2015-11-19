#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
from NNet.SoftmaxLayer import SoftmaxLayer
from NNet.Util import negative_log_likelihood, LearningRateUpdNormalStrategy, \
    defaultGradParameters
from WindowModelBasic import WindowModelBasic
import theano.tensor as T

class WindowModelByWord(WindowModelBasic):

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,
                 numClasses, numEpochs, batchSize=1, c=0.0, charModel=None,
                 learningRateUpdStrategy=LearningRateUpdNormalStrategy(),
                 wordVecsUpdStrategy='normal', networkAct='tanh', norm_coef=1.0):
        #
        # Base class constructor.
        #
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize,
                                  hiddenSize, _lr, numClasses, numEpochs,
                                  batchSize, c, charModel,
                                  learningRateUpdStrategy, False,
                                  wordVecsUpdStrategy, False, networkAct,
                                  norm_coef)

        self.setTestValues = True

        # A camada de saída é um softmax sobre as classes.
        self.softmax = SoftmaxLayer(self.hiddenLayer.getOutput(),
                                    self.hiddenSize,
                                    numClasses)

        # Saída da rede.
        output = self.softmax.getOutput()

        # Training cost function.
        cost = negative_log_likelihood(output, self.y)
        
        #
        # TODO: criar uma forma de integrar a regularização.
        # + regularizationSquareSumParamaters(self.parameters, self.regularizationFactor, self.y.shape[0])
        #

        updates = []

        if charModel != None:
            self.charModel.setCost(cost)
            self.charModel.setUpdates()
            updates += self.charModel.updates

        # Parameters of the ordinary layers (non-structured).
        parameters = self.softmax.getParameters() + self.hiddenLayer.getParameters()

        # Updates of the structured layers.
        updates += self.embedding.getUpdates(cost, self.lr)
        # TODO: considerar a parte estruturada do CharWNN
        
        # Add normalization updates.
        if (self.wordVecsUpdStrategy != 'normal'):
            updates += self.embedding.getNormalizationUpdate(self.wordVecsUpdStrategy, self.norm_coef)
        
        # Adiciona o update padrão dos parâmetros não estruturados.
        updates += defaultGradParameters(cost, parameters, self.lr)
        
        self.cost = cost
        self.updates = updates
        self.paramters = parameters

    def reshapeCorrectData(self, correctData):
        return np.asarray(correctData, dtype=np.int32)
      
    # Esta funcao retorna todos os indices das janelas de palavras  
    def getAllWindowIndexes(self, data):
        allWindowIndexes = []
        
        for idxWord in range(len(data)):
            allWindowIndexes.append(self.getWindowIndexes(idxWord, data))
            
        return np.array(allWindowIndexes, dtype=np.int32)
    
    def confBatchSize(self, inputData):
        numWords = len(inputData)
        
        # Configura o batch size
        if isinstance(self.batchSize, list):
            if sum(self.batchSize) < numWords:
                print "The number of words for training set by batch is smaller than the number of words in inputData"
            else:
                raise Exception("The total number of words in batch exceeds the number of words in inputData")
            
            return np.asarray(self.batchSize, dtype=np.int32);

        num = numWords / self.batchSize  
        arr = np.full(num, self.batchSize, dtype=np.int32)
        if numWords % self.batchSize:
            arr = np.append(arr, numWords % self.batchSize)

        return arr
        # return np.full(numWords/self.batchSize + 1,self.batchSize,dtype=np.int32)
        
    def predict(self, inputData, indexesOfRawWord, unknownDataTest):
      
        self.reloadWindowIds = True  
        y = 0
        
        if self.setTestValues:
            self.testWordWindowIdxs = self.getAllWindowIndexes(inputData)
            
            if self.charModel:
                self.charModel.updateAllCharIndexes(unknownDataTest)
            
                charmodelIdxPos = self.charModel.getAllWordCharWindowIndexes(indexesOfRawWord)
                
                self.testCharWindowIdxs = charmodelIdxPos[0]
                self.testPosMaxByWord = charmodelIdxPos[1]
                self.testNumCharByWord = charmodelIdxPos[2]
                
            self.setTestValues = False    
        
        # self.windowIdxs.set_value(self.testWordWindowIdxs, borrow=True)
        if self.charModel == None:
            
            y_pred = self.softmax.getPrediction()
            pred = theano.function([], [y_pred], givens={self.windowIdxs: self.testWordWindowIdxs})
            y = pred()[0]

        else:

            self.charModel.charWindowIdxs.set_value(self.testCharWindowIdxs, borrow=True)
            self.charModel.posMaxByWord.set_value(self.testPosMaxByWord, borrow=True)
            
            index = T.iscalar("index")
            charIndex = T.iscalar("charIndex")
            step = T.iscalar("step")
            
            
            self.charModel.batchSize.set_value(1)
            
            pred = theano.function(inputs=[index, charIndex, step],
                                    outputs=self.softmax.getPrediction(),
                                    givens={
                                            self.charModel.charWindowIdxs: self.charModel.charWindowIdxs[charIndex:charIndex + step],
                                            self.charModel.posMaxByWord:self.charModel.posMaxByWord[index * self.windowSize:(index + 1) * self.windowSize],
                                            self.windowIdxs: self.windowIdxs[index : index + 1],
                                            
                                    })
            y = []
            j = 0
            for i in range(len(inputData)):
                y.append(pred(i, j, sum(self.testNumCharByWord[i * self.windowSize:(i + 1) * self.windowSize]))[0]);
                j += sum(self.testNumCharByWord[i * self.windowSize:(i + 1) * self.windowSize])
            
            
        return y
        
