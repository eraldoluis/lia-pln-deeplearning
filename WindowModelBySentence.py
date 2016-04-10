#!/usr/bin/env python
# -*- coding: utf-8 -*-

from NNet.SentenceSoftmaxLayer import SentenceSoftmaxLayer
from NNet.SoftmaxLayer import SoftmaxLayer
from NNet.Util import regularizationSquareSumParamaters,\
    LearningRateUpdNormalStrategy
from NNet.Util import negative_log_likelihood, LearningRateUpdNormalStrategy, \
    defaultGradParameters
from WindowModelBasic import WindowModelBasic
import numpy as np
from itertools import chain
import theano.tensor as T
import numpy
import theano

from NNet.HiddenLayer import HiddenLayer

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


    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, wordConvSize, _lr,
                 numClasses, numEpochs, batchSize=1, c=0.0, charModel=None,
                 learningRateUpdStrategy = LearningRateUpdNormalStrategy(),
                 wordVecsUpdStrategy='normal', choice = NeuralNetworkChoiceEnum.COMPLETE,
                 networkAct = 'tanh', norm_coef=1.0, structGrad=True, adaGrad=False,
                 randomizeInput = True, embeddingNotUpdate = [], task='postag', structPrediction = False):
        
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize,
                                  hiddenSize, wordConvSize, _lr, numClasses, numEpochs,
                                  batchSize, c, charModel,
                                  learningRateUpdStrategy, randomizeInput,
                                  wordVecsUpdStrategy, NeuralNetworkChoiceEnum.withoutHiddenLayer(choice),
                                  networkAct, norm_coef, structGrad, adaGrad, task)
    
        self.setTestValues = True
        self.structPrediction = structPrediction
        # Camada: softmax
        
        cost = []
        layers = self.embeddings 

        if self.structPrediction == True:
            #NOT WorKing: viterbi
                        
            if NeuralNetworkChoiceEnum.withoutHiddenLayer(choice):
                if self.task == 'postag':
                    if self.charModel == None:
                        print 'Softmax linked with w2v'
                        self.sentenceSoftmax = SentenceSoftmaxLayer(self.embedding.getOutput(), self.wordSize * self.windowSize, numClasses);
                    else:
                        print 'Softmax linked with w2v and charwv'    
                        self.sentenceSoftmax = SentenceSoftmaxLayer(T.concatenate([self.embedding.getOutput(), self.charModel.getOutput()], axis=1), (self.wordSize + self.charModel.convSize) * self.windowSize , numClasses);
                else:
                    self.sentenceSoftmax = SentenceSoftmaxLayer(self.sentenceFeature, self.wordConvSize, numClasses);
                    
                parameters = self.sentenceSoftmax.getParameters()
                layers = layers + [self.sentenceSoftmax]
            else:
                print 'Softmax linked with hidden'
                
                self.sentenceSoftmax = SentenceSoftmaxLayer(self.hiddenLayer.getOutput(), self.hiddenSize, numClasses);
                parameters = self.sentenceSoftmax.getParameters() + self.hiddenLayer.getParameters()                 
                layers = layers + [ self.hiddenLayer, self.sentenceSoftmax]
                
        else :
            if NeuralNetworkChoiceEnum.withoutHiddenLayer(choice):
                if self.task == 'postag':
                    if self.charModel == None:
                        print 'Softmax linked with w2v'
                        self.sentenceSoftmax = SoftmaxLayer(self.embedding.getOutput(), self.wordSize * self.windowSize, numClasses);
                    else:
                        print 'Softmax linked with w2v and charwv'    
                        self.sentenceSoftmax = SoftmaxLayer(T.concatenate([self.embedding.getOutput(), self.charModel.getOutput()], axis=1), (self.wordSize + self.charModel.convSize) * self.windowSize , numClasses);
                
                else:
                    self.sentenceSoftmax = SoftmaxLayer(self.sentenceFeature, self.wordConvSize, numClasses);
                    
                parameters = self.sentenceSoftmax.getParameters()
                layers = layers + [self.sentenceSoftmax]
            else:
                print 'Softmax linked with hidden'
                          
                self.sentenceSoftmax = SoftmaxLayer(self.hiddenLayer.getOutput(), self.hiddenSize, numClasses);
                
                if self.task == 'sentiment_analysis':
                    parameters = self.sentenceSoftmax.getParameters() + self.sentenceLayer.getParameters() + self.hiddenLayer.getParameters()
                    layers = layers + [self.sentenceLayer, self.hiddenLayer, self.sentenceSoftmax]
                else:
                    parameters = self.sentenceSoftmax.getParameters() + self.hiddenLayer.getParameters()
                    layers = layers + [ self.hiddenLayer, self.sentenceSoftmax]
            
        
        idxToUpdateLayer = filter(lambda k: k not in embeddingNotUpdate , range(0,len(layers)))
        layersToUpdate = [layers[idx] for idx in idxToUpdateLayer]
        
        
        for indexToNotUpdate in embeddingNotUpdate:
            if indexToNotUpdate >= len(self.embeddings):
                raise Exception("Desabilitando updates de layers que não são embeddings")
        
        
        if charModel != None:
            parameters += self.charModel.hiddenLayer.getParameters()
            layers.append(charModel)
            layersToUpdate.append(charModel)
                            
            
        # Custo      
        if structPrediction == True:
            
            logOfSumAllPath = self.sentenceSoftmax.getLogOfSumAllPathY()
            negativeLogLikehood = -(self.sentenceSoftmax.getSumPathY(self.y) - logOfSumAllPath)
            cost =   negativeLogLikehood + regularizationSquareSumParamaters(parameters, self.regularizationFactor, self.y.shape[0]);
        
        else:
            # Saída da rede.
            output = self.sentenceSoftmax.getOutput()
        
            # Training cost function.
            cost = negative_log_likelihood(output, self.y)
            
        # Lists of variables that store the sum of the squared historical 
        # gradients for the parameters of all layers. Since some layers use
        # structured gradients, the historical gradients are also structured
        # and need special treatment. So, we need to store two lists of 
        # historical gradients: one for default gradient parameters and another
        # for structured gradient parameters.
        self.__sumsSqDefGrads = None
        self.__sumsSqStructGrads = None
        if self.isAdaGrad():
            sumsSqDefGrads = []
            for l in  layersToUpdate:
                # Default gradient parameters also follow a default AdaGrad update.
                params = l.getDefaultGradParameters()
                ssgs = []
                for param in params:
                    ssgVals = numpy.zeros(param.get_value(borrow=True).shape,
                                          dtype=theano.config.floatX)
                    ssg = theano.shared(value=ssgVals,
                                        name='sumSqGrads_' + param.name,
                                        borrow=True)
                    ssgs.append(ssg)
                # For default gradient parameters, we do not need to store 
                # parameters or historical gradients separated by layer. We 
                # just store a list of parameters and historical gradients.
                sumsSqDefGrads += ssgs
            self.__sumsSqDefGrads = sumsSqDefGrads
            
            sumsSqStructGrads = []
            for l in  layersToUpdate:
                # Structured parameters also need structured updates for the 
                # historical gradients. These updates are computed by each layer.
                params = l.getStructuredParameters()
                ssgs = []
                for param in params:
                    ssgVals = numpy.zeros(param.get_value(borrow=True).shape,
                                          dtype=theano.config.floatX)
                    ssg = theano.shared(value=ssgVals,
                                        name='sumSqGrads_' + param.name,
                                        borrow=True)
                    ssgs.append(ssg)
                # For structured parameters, we need to store the historical 
                # gradients separated by layer, since the updates of these 
                # variables are performed by each layer.
                sumsSqStructGrads.append(ssgs)
            self.__sumsSqStructGrads = sumsSqStructGrads
        
        # Build list of updates.
        updates = []
        defaultGradParams = []
        
            
        # Get structured updates and default-gradient parameters from all layers.
        for (idx, l) in enumerate(layersToUpdate):
            # Structured updates (embeddings, basically).
            ssgs = None
            if self.isAdaGrad():
                ssgs = self.__sumsSqStructGrads[idx]
            updates += l.getUpdates(cost, self.lr, ssgs)
            # Default gradient parameters (all the remaining).
            defaultGradParams += l.getDefaultGradParameters()
        
        # Add updates for default-gradient parameters.
        updates += defaultGradParameters(cost, defaultGradParams,
                                         self.lr, self.__sumsSqDefGrads)
        
        # Add normalization updates.
        if (self.wordVecsUpdStrategy != 'normal'):
            updates += self.embedding.getNormalizationUpdate(self.wordVecsUpdStrategy, self.norm_coef)
        if (self.charModel and self.charModel.charVecsUpdStrategy != 'normal'):
            updates += self.charModel.getNormalizationUpdate(self.charModel.charVecsUpdStrategy, self.norm_coef)

        # Store cost and updates to be used in the training function.        
        self.cost = cost
        self.updates = updates
        
                
    def reshapeCorrectData(self,correctData):
        if self.task == 'postag':
            return np.fromiter(chain.from_iterable(correctData),dtype=np.int32)
        else:
            return np.asarray(correctData, dtype=np.int32)
            
    
    def getAllWindowIndexes(self, data):
        allWindowIndexes = [];
        self.sentencesSize = [];
        
        for idxSentence in range(len(data)):
            for idxWord in range(len(data[idxSentence])):
                allWindowIndexes.append(self.getWindowIndexes(idxWord, data[idxSentence]))
            
            self.sentencesSize.append(len(data[idxSentence]))
        
        return np.array(allWindowIndexes, dtype=np.int32)
    
    def confBatchSize(self,data):
        # Configura o batch size
        return np.asarray(self.sentencesSize,dtype=np.int32)
        
    
    def predict(self, inputData, indexesOfRawWord, unknownDataTest):
        
        self.reloadWindowIds = True
        
        if self.setTestValues:
            # We need to generate test data in the format expected by the NN.
            # That is, list of word- and character-level features.
            # But this needs to be done only once, even when evaluation is
            # performed after several epochs.
            
            #self.testWordWindowIdxs = self.getAllWindowIndexes(inputData)
            
            self.testWordWindowIdxs = theano.shared(self.getAllWindowIndexes(inputData),
                                   name="window_shared",
                                   borrow=True)
            
            if self.charModel:
                self.charModel.updateAllCharIndexes(unknownDataTest)
                #self.testCharWindowIdxs = self.charModel.getAllWordCharWindowIndexes(indexesOfRawWord)
                self.testCharWindowIdxs = theano.shared(self.charModel.getAllWordCharWindowIndexes(indexesOfRawWord), borrow=True)
                
                
            self.testBatchSize = self.confBatchSize(inputData)
            
            self.testBeginBlock = []    
            pos = 0
        
            for v in self.testBatchSize:
                self.testBeginBlock.append(pos)
                pos += v
        
            self.setTestValues = False
                
        
        predicts = []
        
               
        testBatchIndex = T.iscalar("testBatchIndex")
        testIndex = T.iscalar("testIndex")
        
        # Input of the word-level embedding.
        givens = {self.windowIdxs : self.testWordWindowIdxs[testIndex : testIndex + testBatchIndex]}
        if self.charModel:
            # Input of the character-level embedding.
            givens[self.charModel.charWindowIdxs] = self.testCharWindowIdxs[testIndex : testIndex + testBatchIndex ]
            
        if self.structPrediction == False:
            # Predicted values.
            y_pred = self.sentenceSoftmax.getPrediction()
                
        else:
            # Predicted values.
            y_pred = self.sentenceSoftmax.getPrediction(testBatchIndex)
            
        # Prediction function.
        pred = theano.function([testIndex, testBatchIndex], y_pred, givens=givens)
        
        for idx in range(len(self.testBatchSize)):     
            
            predicts.append(pred( self.testBeginBlock[idx], self.testBatchSize[idx]))
            
                         
        return np.asarray(predicts);
