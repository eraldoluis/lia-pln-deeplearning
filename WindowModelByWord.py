#!/usr/bin/env python
# -*- coding: utf-8 -*-
    
import itertools
import numpy as np
import theano
from NNet.SoftmaxLayer import SoftmaxLayer
from NNet.Util import negative_log_likelihood, LearningRateUpdNormalStrategy, \
    defaultGradParameters
from WindowModelBasic import WindowModelBasic
import numpy

class WindowModelByWord(WindowModelBasic):
    
    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, _lr,
                 numClasses, numEpochs, batchSize=1, c=0.0, charModel=None,
                 learningRateUpdStrategy=LearningRateUpdNormalStrategy(),
                 wordVecsUpdStrategy='normal', networkAct='tanh', norm_coef=1.0,
                 structGrad=True, adaGrad=False,randomizeInput = True,embeddingNotUpdate = [], task='postag'):
        #
        # Base class constructor.
        #
        WindowModelBasic.__init__(self, lexicon, wordVectors, windowSize,
                                  hiddenSize, 0, _lr, numClasses, numEpochs,
                                  batchSize, c, charModel,
                                  learningRateUpdStrategy, randomizeInput,
                                  wordVecsUpdStrategy, False, networkAct,
                                  norm_coef, structGrad, adaGrad, task, False)
        
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
        layers = self.embeddings + [self.hiddenLayer, self.softmax]
        idxToUpdateLayer = filter(lambda k: k not in embeddingNotUpdate , range(0,len(layers)))
        layersToUpdate = [layers[idx] for idx in idxToUpdateLayer]
        
        
        for indexToNotUpdate in embeddingNotUpdate:
            if indexToNotUpdate >= len(self.embeddings):
                raise Exception("Desabilitando updates de layers que não são embeddings")
              
        if charModel:
            layers.append(charModel)
            layersToUpdate.append(charModel)
        
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

    def reshapeCorrectData(self, correctData):
        return np.asarray(correctData, dtype=np.int32)
    
    # Esta funcao retorna todos os indices das janelas de palavras  
    def getAllWindowIndexes(self, data):
        allWindowIndexes = []
        
        for idxWord in xrange(len(data)):
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
        else:
            if self.batchSize > 0 :
                num = numWords / self.batchSize  
                arr = np.full(num, self.batchSize, dtype=np.int32)
                if numWords % self.batchSize:
                    arr = np.append(arr, numWords % self.batchSize)
            else:
                arr = [numWords]
        
        return arr
        # return np.full(numWords/self.batchSize + 1,self.batchSize,dtype=np.int32)
        
    def predict(self, inputData, indexesOfRawWord, unknownDataTest):
        
        self.reloadWindowIds = True
        
        if self.setTestValues:
            # We need to generate test data in the format expected by the NN.
            # That is, list of word- and character-level features.
            # But this needs to be done only once, even when evaluation is
            # performed after several epochs.
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

    def pred_y_given_x(self, inputData, indexesOfRawWord, unknownDataTest):
        
        self.reloadWindowIds = True
        
        if self.setTestValues:
            # We need to generate test data in the format expected by the NN.
            # That is, list of word- and character-level features.
            # But this needs to be done only once, even when evaluation is
            # performed after several epochs.
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
        y_pred_y_given_x = self.softmax.getOutput()
        # Prediction function.
        pred = theano.function([], y_pred_y_given_x, givens=givens)
        # Return the predicted values.
        return pred()
