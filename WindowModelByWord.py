#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
from NNet.SoftmaxLayer import SoftmaxLayer
from NNet.Util import negative_log_likelihood, LearningRateUpdNormalStrategy, \
    defaultGradParameters
from WindowModelBasic import WindowModelBasic

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
        
        # List of layers.
        layers = [self.embedding, self.hiddenLayer, self.softmax]
        if charModel:
            layers.append(charModel)
        
        # Build list of updates.
        updates = []
        defaultGradParams = []
        
        for l in layers:
            # Structured updates (embeddings, basically).
            updates += l.getUpdates(cost, self.lr)
            # Default gradient parameters (all the remaining).
            defaultGradParams += l.getDefaultGradParameters()
        
        # Add default updates.
        updates += defaultGradParameters(cost, defaultGradParams, self.lr)
        
        # Add normalization updates.
        if (self.wordVecsUpdStrategy != 'normal'):
            updates += self.embedding.getNormalizationUpdate(self.wordVecsUpdStrategy, self.norm_coef)
        if (self.charModel.charVecsUpdStrategy != 'normal'):
            updates += self.charModel.getNormalizationUpdate(self.charModel.charVecsUpdStrategy, self.norm_coef)

        # Store cost and updates to be used in the training function.        
        self.cost = cost
        self.updates = updates

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
        
        if self.setTestValues:
            # We need to generate test data in the format waited by the NN.
            # That is, list of word- and character-level features.
            # But this needs to be done only once, even when evaluation is
            # performed along the training process.
            self.testWordWindowIdxs = self.getAllWindowIndexes(inputData)
            
            if self.charModel:
                self.charModel.updateAllCharIndexes(unknownDataTest)
                self.testCharWindowIdxs = self.charModel.getAllWordCharWindowIndexes(indexesOfRawWord)
            
            self.setTestValues = False    
        
        # Input of the word-level embedding.
        givens = {self.windowIdxs : self.testWordWindowIdxs}
        if self.charModel:
            # Input of the character-level embedding.
            givens[self.charModel.charWindowIdxs] = self.testCharWindowIdxs
        # Predicted values.
        y_pred = self.softmax.getPrediction()
        # Prediction function.
        pred = theano.function([], y_pred, givens=givens)
        # Return the predicted values.
        return pred()
