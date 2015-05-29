#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from NNet.HiddenLayer import HiddenLayer
from NNet.WortToVectorLayer import WordToVectorLayer
#from theano.tensor.nnet.nnet import softmax
#from NNet.SoftmaxLayer import SoftmaxLayer
#from NNet.Util import negative_log_likelihood, regularizationSquareSumParamaters
#from WindowModelBasic import WindowModelBasic
from NNet.Util import defaultGradParameters,WeightTanhGenerator


class CharWNN():

    def __init__(self,inputData, charcon, charVectors , charWindowSize,wordWindowSize , convSize, _lr,numClasses,numEpochs, batchSize=1, c=0.0):
        
        
        self.Cv = theano.shared(name='charVecs',
                                value=np.asarray(charVectors.getWordVectors(), dtype=theano.config.floatX),
                                borrow=True)
        self.charSize = charVectors.getLenWordVector()
        self.lr = _lr
        self.charWindowSize = charWindowSize
        self.wordWindowSize = wordWindowSize
        self.startSymbol = charcon.getLexiconIndex("<s>")
        self.endSymbol = charcon.getLexiconIndex("</s>")
        self.numClasses = numClasses
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.update = None
        self.cost = None
        self.regularizationFactor = theano.shared(c)
        self.convSize = convSize
        self.weightTanhGenerator = WeightTanhGenerator()
        self.wordIdx = 0
        self.curIdx = 0
        #self.A = np.asarray(
        #        self.weightTanhGenerator.generateWeight(numClasses+1,numClasses),
        #        dtype=theano.config.floatX
        #    )
        
        #self.initWithBasicLayers()
           
        # Camada: allWindowChar terá o índice das janelas dos caracteres de todas as palavras.
        self.windowAllIdxs = theano.shared(self.getAllWindowIndexes(inputData) ,name="windowAllIdxs",borrow=True)
        
        #Vetor: numCharByWord com o número de caracteres por palavra
        self.numCharByWord = theano.shared(np.asarray(self.numCharByWord),'numCharByWord',int)
                
        # Camada: lookup table.
        #self.wordToVector = WordToVectorLayer(self.windowAllIdxs,self.Cv, self.charSize, True)
        self.wordToVector = WordToVectorLayer(self.windowAllIdxs[self.curIdx:self.curIdx+self.numCharByWord[self.wordIdx]],self.Cv, self.charSize, True)
        
        # Camada: hidden layer com a função Tanh como função de ativaçãos
        self.hiddenLayer = HiddenLayer(self.wordToVector.getOutput(),self.charSize * self.charWindowSize , self.convSize);

        # inicialização da função que returna representação a nivel de caracteres        
        numWor = self.batchSize *  self.wordWindowSize
        
        curIdx = T.scalar('curIdx',dtype='int64') 
        wordIdx = T.scalar('wordIdx',dtype='int64')
        charEmbedding = T.zeros((numWor,self.convSize))
        a = T.arange(0,numWor)
        
        def maxByWord(ind,wordIdx,charEmbedding,curIdx,dot):
            numChar = self.numCharByWord[wordIdx]
            #charEmbedding =  T.set_subtensor(charEmbedding[ind], T.max(dot[curIdx:curIdx+numChar], 0))
            charEmbedding =  T.set_subtensor(charEmbedding[ind], T.max(dot, 0))
            curIdx = curIdx + numChar
            wordIdx = wordIdx + 1
            return [wordIdx,charEmbedding,curIdx]
        
        [s,r,i], updates = theano.scan(fn= maxByWord,
                                       sequences = a,
                                       outputs_info = [wordIdx,charEmbedding,curIdx],
                                       non_sequences =self.hiddenLayer.getOutput(),
                                       n_steps = numWor)
        
        # Este o vector que será concatenato com o do wordVector
        ans = r[-1].reshape((batchSize,self.wordWindowSize * self.convSize))
        posChar = i[-1]
        posWord = s[-1]
        #self.windowIdxs = self.windowAllIdxs[curIdx:curIdx+self.numCharByWord[wordIdx]]
        self.getCharEmbeddingOfWindow = theano.function(inputs=[wordIdx,curIdx], outputs=[posWord,ans,posChar],
                                                        #givens={
                                                        #        self.windowIdxs:self.windowAllIdxs[curIdx:curIdx+self.numCharByWord[wordIdx]]
                                                        #        }
                                                        )
                                                        
        
        # Gradiente dos pesos e do bias
         
        #updates += self.getAUpdate(self.cost,self.lr);
        #self.setUpdates(updates)
    
    
    
    def initWithBasicLayers(self):
        # Camada: word window.
        #self.windowIdxs = theano.shared(value=np.zeros((1,self.charWindowSize),dtype="int64"),name="windowIdxs")
        
        
        # Camada: lookup table.
        self.wordToVector = WordToVectorLayer(self.windowIdxs,self.Cv, self.charSize, True)
        
        # Camada: hidden layer com a função Tanh como função de ativaçãos
        self.hiddenLayer = HiddenLayer(self.wordToVector.getOutput(),self.charSize * self.charWindowSize , self.convSize);
        
        
    def getAllWindowIndexes(self, data):
        allWindowIndexes = []
        self.numCharByWord = []
        
        for idxWord in range(len(data)):
            self.numCharByWord.append(len(data[idxWord]))
            for idxChar in range(len(data[idxWord])):
                allWindowIndexes.append(self.getWindowIndexes(idxChar, data[idxWord]))
            
        return np.array(allWindowIndexes);

    def getWindowIndexes(self, idxWord, data):
        lenData = len(data)
        windowNums = []
        contextSize = int(np.floor((self.charWindowSize - 1) / 2))
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
        raise NotImplementedError();
     
    def setUpdates(self):
        updates = self.hiddenLayer.getUpdate(self.cost, self.lr);
        
        updates += self.wordToVector.getUpdate(self.cost, self.lr);
        self.updates = updates
        
    def setCost(self,cost):
        self.cost = cost
        
    def setLr(self,_lr):
        self.lr = _lr
    
    def setIndexNull(self):
        self.wordIdx = 0
        self.curIdx = 0    
    
    def getAUpdate(self,cost,learningRate):
        return defaultGradParameters(cost,self.A,learningRate);
    
    def getOutput(self):
        [self.wordIdx,ans,self.curIdx] = self.getCharEmbeddingOfWindow(self.wordIdx,self.curIdx);
        return ans

