#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NNet.SentenceSoftmaxLayer import SentenceSoftmaxLayer
from NNet.Util import regularizationSquareSumParamaters,\
    LearningRateUpdNormalStrategy
from WindowModelBasic import WindowModelBasic
import numpy as np
from itertools import chain

class WindowModelBySentence(WindowModelBasic):

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1, c=0.0,
                 charModel=None,learningRateUpdStrategy = LearningRateUpdNormalStrategy()):
        
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize, hiddenSize, _lr, 
                                  numClasses, numEpochs, batchSize, c,charModel,learningRateUpdStrategy,True)
    
        
        # Camada: softmax
        self.sentenceSoftmax = SentenceSoftmaxLayer(self.hiddenLayer.getOutput(), self.hiddenSize, numClasses);
        parameters = self.sentenceSoftmax.getParameters() + self.hiddenLayer.getParameters()
            
        if charModel == None:
            
            # Custo      
            logOfSumAllPath = self.sentenceSoftmax.getLogOfSumAllPathY()
            negativeLogLikehood = -(self.sentenceSoftmax.getSumPathY(self.y) - logOfSumAllPath)
            cost =   negativeLogLikehood + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
            
            # Gradiente dos pesos e do bias
            updates = self.hiddenLayer.getUpdate(cost, self.lr);
            
        else:
    
            parameters += self.charModel.hiddenLayer.getParameters()
            
            # Custo      
            logOfSumAllPath = self.sentenceSoftmax.getLogOfSumAllPathY()
            negativeLogLikehood = -(self.sentenceSoftmax.getSumPathY(self.y) - logOfSumAllPath)
            cost =   negativeLogLikehood + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
                
            self.charModel.setCost(cost)
            self.charModel.setUpdates()
            
            updates = self.hiddenLayer.getUpdate(cost, self.lr);
            updates += self.charModel.updates
            
        
        updates += self.sentenceSoftmax.getUpdate(cost, self.lr);
        updates += self.wordToVector.getUpdate(cost, self.lr);
         
        self.setCost(cost)
        self.setUpdates(updates)
        
    def reshapeCorrectData(self,correctData):
        return np.fromiter(chain.from_iterable(correctData),dtype=int)
    
    def getAllWindowIndexes(self, data):
        allWindowIndexes = [];
        self.sentencesSize = [];
        
        for idxSentence in range(len(data)):
            for idxWord in range(len(data[idxSentence])):
                allWindowIndexes.append(self.getWindowIndexes(idxWord, data[idxSentence]))
            
            self.sentencesSize.append(len(data[idxSentence]))
        
        return np.array(allWindowIndexes);
    
    def confBatchSize(self,data):
        # Configura o batch size
        return np.asarray(self.sentencesSize,dtype=np.int64)
        
    
    def predict(self, inputData,inputDataRaw,unknownDataTest):
        windowSetences = self.getAllWindowIndexes(inputData);
        
        predicts = []
        index = 0
        indexSentence = 0
        self.reloadWindowIds = True
        
        if self.charModel==None:
        
            while index < len(windowSetences):
                self.windowIdxs.set_value(windowSetences[index:index + self.sentencesSize[indexSentence]],borrow=True)
                
                predicts.append(self.sentenceSoftmax.predict(self.sentencesSize[indexSentence]))
                
                
                index += self.sentencesSize[indexSentence]
                indexSentence += 1
        
        else:
            self.charModel.updateAllCharIndexes(unknownDataTest)
            
            charmodelIdxPos = self.charModel.getAllWordCharWindowIndexes(inputDataRaw)
            charWindow = charmodelIdxPos[0]
            posMax = charmodelIdxPos[1]
            numCharByWord = charmodelIdxPos[2]
            
            numCharsInSentence = self.charModel.confBatchSize(numCharByWord,[])
            
            
            
            charIndex = 0
            while index < len(windowSetences):
                self.windowIdxs.set_value(windowSetences[index:index + self.sentencesSize[indexSentence]],borrow=True)
                self.charModel.charWindowIdxs.set_value(charWindow[charIndex:charIndex+numCharsInSentence[indexSentence]],borrow=True)
                self.charModel.posMaxByWord.set_value(posMax[index*self.windowSize:(index+self.sentencesSize[indexSentence])*self.windowSize],borrow=True)
                self.charModel.batchSize.set_value(self.sentencesSize[indexSentence])
                
                predicts.append(self.sentenceSoftmax.predict(self.sentencesSize[indexSentence]))
                
                charIndex += numCharsInSentence[indexSentence]
                index += self.sentencesSize[indexSentence]
                indexSentence += 1
                
        
        return np.asarray(predicts);
