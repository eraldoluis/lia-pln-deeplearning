#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NNet.SentenceSoftmaxLayer import SentenceSoftmaxLayer
from NNet.Util import regularizationSquareSumParamaters,\
    LearningRateUpdNormalStrategy
from WindowModelBasic import WindowModelBasic
import numpy as np
from itertools import chain
import theano.tensor as T

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


    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,numClasses,numEpochs, batchSize=1, c=0.0,
                 charModel=None,learningRateUpdStrategy = LearningRateUpdNormalStrategy(),wordVecsUpdStrategy='normal', choice = NeuralNetworkChoiceEnum.COMPLETE,networkAct = 'tanh',norm_coef=1.0,
                 wvNotUpdate = []):
        
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize, hiddenSize, _lr, 
                                  numClasses, numEpochs, batchSize, c,charModel,learningRateUpdStrategy,True,wordVecsUpdStrategy,NeuralNetworkChoiceEnum.withoutHiddenLayer(choice),networkAct,norm_coef)
    
        self.setTestValues = True
        # Camada: softmax

        
        if NeuralNetworkChoiceEnum.withoutHiddenLayer(choice):
            
            if self.charModel == None:
                print 'Softmax linked with w2v'
                self.sentenceSoftmax = SentenceSoftmaxLayer(self.concatenateOutputWv, self.wordSize * self.windowSize, numClasses);
            else:
                print 'Softmax linked with w2v and charwv'    
                self.sentenceSoftmax = SentenceSoftmaxLayer(T.concatenate([self.concatenateOutputWv, self.charModel.getOutput()], axis=1), (self.wordSize + self.charModel.convSize) * self.windowSize , numClasses);
            
            parameters = self.sentenceSoftmax.getParameters()
        else:
            print 'Softmax linked with hidden'
            self.sentenceSoftmax = SentenceSoftmaxLayer(self.hiddenLayer.getOutput(), self.hiddenSize, numClasses);
            parameters = self.sentenceSoftmax.getParameters() + self.hiddenLayer.getParameters()
            
        # Custo
        
        if charModel != None:
            parameters += self.charModel.hiddenLayer.getParameters()
            
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
            
            isToUpdateWv = [True for wvLayer in self.wordToVector]
            
            for indexToNotUpdate in wvNotUpdate:
                isToUpdateWv[indexToNotUpdate - 1] = False
            
            for i in range(len(self.wordToVector)):
                if isToUpdateWv[i]:
                    updates += self.wordToVector[i].getUpdate(cost, self.lr); 
        else:
            print 'Without update vector'
            
        if charModel != None:
               
            self.charModel.setCost(cost)
            self.charModel.setUpdates()
            
            updates += self.charModel.updates  
        
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
        
        
        predicts = []
        index = 0
        indexSentence = 0
        self.reloadWindowIds = True
        
        if self.setTestValues:
            
            self.testSentenceWindowIdxs = self.getAllWindowIndexes(inputData)
            
    
            if self.charModel:
                self.charModel.updateAllCharIndexes(unknownDataTest)
                
            
                charmodelIdxPos = self.charModel.getAllWordCharWindowIndexes(inputDataRaw)
                self.testCharWindowIdxs = charmodelIdxPos[0]
                self.testPosMaxByWord = charmodelIdxPos[1]
                self.testNumCharBySentence = charmodelIdxPos[2]
                
                
            self.setTestValues = False    
        
        if self.charModel==None:
        
            while index < len(self.testSentenceWindowIdxs):
                self.windowIdxs.set_value(self.testSentenceWindowIdxs[index:index + self.sentencesSize[indexSentence]],borrow=True)
                
                predicts.append(self.sentenceSoftmax.predict(self.sentencesSize[indexSentence]))
                
                
                index += self.sentencesSize[indexSentence]
                indexSentence += 1
        
        else:
                      
            
            charIndex = 0
            while index < len(self.testSentenceWindowIdxs):
                step = sum(self.testNumCharBySentence[indexSentence])
   
                self.windowIdxs.set_value(self.testSentenceWindowIdxs[index:index + self.sentencesSize[indexSentence]],borrow=True)
                self.charModel.charWindowIdxs.set_value(self.testCharWindowIdxs[charIndex:charIndex+step],borrow=True)
                self.charModel.posMaxByWord.set_value(self.testPosMaxByWord[index*self.windowSize:(index+self.sentencesSize[indexSentence])*self.windowSize],borrow=True)
                self.charModel.batchSize.set_value(self.sentencesSize[indexSentence])
                
                predicts.append(self.sentenceSoftmax.predict(self.sentencesSize[indexSentence]))
                
                charIndex += step
                index += self.sentencesSize[indexSentence]
                indexSentence += 1
                
        
        return np.asarray(predicts);
