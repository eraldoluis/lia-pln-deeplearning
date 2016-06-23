#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.compile
import theano.tensor as T
from NNet.HiddenLayer import HiddenLayer
from NNet.EmbeddingLayer import EmbeddingLayer
from NNet.Util import LearningRateUpdNormalStrategy
import time
import random
import logging

class WindowModelBasic:
    startSymbolStr = u'<s>'
    endSymbolStr = u'</s>'
    
    @staticmethod
    def setStartSymbol(startSymbol):
        WindowModelBasic.startSymbolStr = startSymbol

    @staticmethod
    def setEndSymbol(endSymbol):
        WindowModelBasic.endSymbolStr = endSymbol
        
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_WindowModelBasic__log']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
        self.__log = logging.getLogger(__name__)

    def __init__(self, lexicon, wordVectors , windowSize, hiddenSize, wordConvSize, _lr,
                 numClasses, numEpochs, batchSize=1.0, c=0.0, charModel=None,
                 learningRateUpdStrategy=LearningRateUpdNormalStrategy(),
                 randomizeInput=False, wordVecsUpdStrategy='normal',
                 withoutHiddenLayer=False, networkAct='tanh', norm_coef=1.0,
                 structGrad=True, adaGrad=False, task='postag', senLayerWithAct=False):
        # Logging object.
        self.__log = logging.getLogger(__name__)

        self.Wv = []
        
        if not isinstance(wordVectors, list):
            wordVectors = [wordVectors]
        
        a = 1
        self.wordSize = 0
        
        for wv in wordVectors:
            self.Wv.append(theano.shared(name='wordVec' + str(a),
                                value=np.asarray(wv.getWordVectors(), dtype=theano.config.floatX),
                                borrow=True))
            
            self.wordSize += wv.getLenWordVector()
            
            a+=1
        
        
        self.lrValue = _lr
        self.lr = T.dscalar('lr')
        self.learningRateUpdStrategy = learningRateUpdStrategy;
        self.hiddenSize = hiddenSize;
        self.windowSize = windowSize
        
        self.startSymbol = lexicon.getLexiconIndex(WindowModelBasic.startSymbolStr)
        self.endSymbol = lexicon.getLexiconIndex(WindowModelBasic.endSymbolStr)
        self.numClasses = numClasses
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.cost = None
        self.update = None
        self.regularizationFactor = theano.shared(c)
        self.wordConvSize = wordConvSize
        
        self.wordVecsUpdStrategy = wordVecsUpdStrategy
        self.charModel = charModel
        self.isToRandomizeInput = randomizeInput
        self.numCharByWord = theano.shared(np.asarray([0]), 'numCharByWord', int)
        
        self.networkAct = networkAct
        self.norm_coef = norm_coef
        
        self.__structGrad = structGrad
        self.__adaGrad = adaGrad
        
        self.task = task
        self.senLayerWithAct = senLayerWithAct
        
        self.initWithBasicLayers(withoutHiddenLayer)
        
        self.listeners = []
        
    def addListener(self, listener):
        self.listeners.append(listener)
        
    def setCost(self, cost):
        self.cost = cost
    
    def setUpdates(self, updates):
        self.updates = updates
    
    def initWithBasicLayers(self, withoutHiddenLayer):
        # Variável que representa uma entrada para a rede. A primeira camada é 
        # a camada de embedding que recebe um batch de janelas de palavras.
        # self.windowIdxs = theano.shared(value=np.zeros((1, self.windowSize), dtype="int64"), name="windowIdxs")
        self.windowIdxs = T.imatrix("windowIdxs")
        
        # Variável que representa a saída esperada para cada exemplo do (mini) batch.
        self.y = T.ivector("y")
        
        # Word embedding layer
        # self.embedding = EmbeddingLayer(self.windowIdxs, self.Wv, self.wordSize, True,self.wordVecsUpdStrategy,self.norm_coef)
#         self.embedding = EmbeddingLayer(examples=self.windowIdxs,
#                                         embedding=self.Wv,
#                                         structGrad=self.__structGrad)
        
        # Camada: lookup table.
        self.embeddings = []
        
        for Wv in self.Wv:
            embedding = EmbeddingLayer(examples=self.windowIdxs,
                                embedding=Wv,
                                structGrad=self.__structGrad)
            self.embeddings.append(embedding)
        
        self.concatenateEmbeddings = T.concatenate( [embeddingLayer.getOutput() for embeddingLayer in self.embeddings],axis=1)
        
        # Camada: sentence layer 
        if self.task == 'sentiment_analysis':
            print 'With Sentence Layer'
            
            act = self.networkAct if self.senLayerWithAct == True else None
            if self.charModel == None:
                self.sentenceLayer  =   HiddenLayer(self.concatenateEmbeddings,
                                                        self.wordSize * self.windowSize,
                                                        self.wordConvSize,
                                                        activation=act)
                    
            else:
                
                self.sentenceLayer  =   HiddenLayer(T.concatenate([self.concatenateEmbeddings,
                                                                       self.charModel.getOutput()],
                                                                      axis=1),
                                                        (self.wordSize + self.charModel.convSize) * self.windowSize,
                                                        self.wordConvSize,
                                                        activation=act)
                                
            mm     =   T.max(self.sentenceLayer.getOutput(), axis=0)

            self.sentenceFeature = mm.reshape((1, self.wordConvSize))
            


        # Camada: hidden layer com a função Tanh com função de ativação
        if not withoutHiddenLayer:
            print 'With Hidden Layer'
            
            if self.task == 'postag':
                if self.charModel == None:
                
                    self.hiddenLayer    =   HiddenLayer(self.concatenateEmbeddings,
                                                        self.wordSize * self.windowSize,
                                                        self.hiddenSize,
                                                        activation=self.networkAct)
                                    
                else:
                
                    self.hiddenLayer    =   HiddenLayer(T.concatenate([self.concatenateEmbeddings,
                                                                       self.charModel.getOutput()],
                                                                      axis=1),
                                                        (self.wordSize + self.charModel.convSize) * self.windowSize,
                                                        self.hiddenSize,
                                                        activation=self.networkAct)
            else:
                            
                self.hiddenLayer    =   HiddenLayer(self.sentenceFeature,
                                                    self.wordConvSize,
                                                    self.hiddenSize,
                                                    activation=self.networkAct)

                    
        else:
            print 'Without Hidden Layer'
    
    def getAllWindowIndexes(self, data):
        raise NotImplementedError();
    
    def reshapeCorrectData(self, correctData):
        raise NotImplementedError();

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
    
    def confBatchSize(self, inputData):
        raise NotImplementedError();
    
    def train(self, inputData, correctData, indexesOfRawWord, folder=None, model_description=None, debug_mode=False):
        # TODO: test
#         n = 1000
#         inputData = inputData[:n]
#         correctData = correctData[:n]
#         indexesOfRawWord = indexesOfRawWord[:n]
        
        log = self.__log

        # Matrix with training data.
        # windowIdxs = theano.shared(self.getAllWindowIndexes(inputData), borrow=True)
        log.info("Generating all training windows (training input)...")
        windowIdxs = theano.shared(self.getAllWindowIndexes(inputData),
                                   name="x_shared",
                                   borrow=True)

        # Correct labels.
        y = theano.shared(self.reshapeCorrectData(correctData),
                          name="y_shared",
                          borrow=True)
        
        
#===============================================================================
#         numExs = len(inputData)
#         if self.batchSize > 0:
#             numBatches = numExs / self.batchSize
#             if numBatches <= 0:
#                 numBatches = 1
#                 self.batchSize = numExs
#             elif numExs % self.batchSize > 0:
#                 numBatches += 1
#         else:
#             numBatches = 1
#             self.batchSize = numExs
#         
# 
#         log.info("Training with %d examples using %d mini-batches (batchSize=%d)..." % (numExs, numBatches, self.batchSize))
#===============================================================================
        
        
              
        # This is the index of the batch to be trained.
        # It is multiplied by self.batchSize to provide the correct slice
        # for each training iteration.
        batchIndex = T.iscalar("batchIndex")
        index = T.iscalar("index")
        senIndex = T.iscalar("senIndex")
        batchesSize = self.confBatchSize(inputData)
        numBatches = len(batchesSize)
        
        self.beginBlock = []    
        pos = 0
        
        for v in batchesSize:
            self.beginBlock.append(pos)
            pos += v
        
        # Word-level training data (input and output).    
        # Character-level training data.
        
        if self.task == 'postag':
            
            outputs = [self.cost]
            
            if self.charModel:
                if debug_mode:
                    outputs += [self.charModel.hiddenLayer.getOutput(),self.charModel.getOutput()]
                
                charInput = theano.shared(self.charModel.getAllWordCharWindowIndexes(indexesOfRawWord), borrow=True)
                givens = {
                          self.windowIdxs: windowIdxs[ index : index + batchIndex ],
                          self.y: y[ index : index + batchIndex],
                          self.charModel.charWindowIdxs : charInput[ index : index + batchIndex ]
                         }
            else:
                givens = {
                          self.windowIdxs: windowIdxs[ index : index + batchIndex ],
                          self.y: y[ index : index + batchIndex ]
                         }
            
            if debug_mode:
                outputs += [self.concatenateEmbeddings]
                
                if self.separateSentence:
                    outputs += [self.sentenceSoftmax.getPrediction()]
                else:
                    outputs += [self.softmax.getPrediction()]
            # Train function.
            train = theano.function(inputs=[index, batchIndex, self.lr],
                                    outputs=outputs,
                                    updates=self.updates,
                                    givens=givens)
            
            
        elif self.task == 'sentiment_analysis':
            
            outputs = [self.cost]
            
            if self.charModel:
                if debug_mode:
                    outputs += [self.charModel.hiddenLayer.getOutput(),self.charModel.getOutput()]
                
                charInput = theano.shared(self.charModel.getAllWordCharWindowIndexes(indexesOfRawWord), borrow=True)
                givens = {
                          self.windowIdxs: windowIdxs[ index : index + batchIndex ],
                          self.y: y[ senIndex : senIndex + 1],
                          self.charModel.charWindowIdxs : charInput[ index : index + batchIndex ]
                         }
            else:
                givens = {
                          self.windowIdxs: windowIdxs[ index : index + batchIndex ],
                          self.y: y[ senIndex : senIndex + 1]
                         }
            if debug_mode:
                outputs += [self.concatenateEmbeddings, self.sentenceLayer.getOutput(), self.sentenceFeature,
                            self.hiddenLayer.getOutput(), self.sentenceSoftmax.getPrediction()]      
            
            # Train function.
            train = theano.function(inputs=[senIndex, index, batchIndex, self.lr],
                                    outputs=outputs,
                                    updates=self.updates,
                                    givens=givens)
            
                
               
        # Indexes of all training (mini) batches.
        idxList = range(numBatches)
        
        for epoch in range(1, self.numEpochs + 1):
            print 'Training epoch ' + str(epoch) + '...'
            
            if self.isToRandomizeInput:
                # Shuffle order of mini-batches for this epoch.
                random.shuffle(idxList)
            
            # Compute current learning rate.
            lr = self.learningRateUpdStrategy.getCurrentLearninRate(self.lrValue, epoch)
            
            t1 = time.time()
            
            if (epoch-1)%5 == 0: 
                
                charhidden_value = []
                charmodel_value = []
                wordmodel_value = []
                senthidden_value = []
                sentmodel_value = []
                hidden_value = []
                softmax_value = []
            # Train each mini-batch.
            for idx in idxList:
                if self.task == 'postag':
                    saida = train( self.beginBlock[idx], batchesSize[idx], lr)
                    
                else:
                    saida = train( idx, self.beginBlock[idx], batchesSize[idx], lr)
                
                if debug_mode:
                    
                    if self.charModel:
                        
                        charhidden_value = np.append(charhidden_value, saida[1].flatten())
                        charmodel_value = np.append(charmodel_value, saida[2].flatten())
                        wordmodel_value = np.append(wordmodel_value, saida[3].flatten())
                        if self.task == 'sentiment_analysis':
                            senthidden_value = np.append(senthidden_value, saida[4].flatten())
                            sentmodel_value = np.append(sentmodel_value, saida[5].flatten())
                            hidden_value = np.append(hidden_value, saida[6].flatten())
                            softmax_value = np.append(softmax_value, saida[7].flatten())
                        else:
                            hidden_value = np.append(hidden_value, saida[4].flatten())
                            softmax_value = np.append(softmax_value, saida[5].flatten())
                        
                    else:
                        wordmodel_value = np.append(wordmodel_value, saida[1].flatten())
                        if self.task == 'sentiment_analysis':
                            senthidden_value = np.append(senthidden_value, saida[2].flatten())
                            sentmodel_value = np.append(sentmodel_value, saida[3].flatten())
                            hidden_value = np.append(hidden_value, saida[4].flatten())
                            softmax_value = np.append(softmax_value, saida[5].flatten())
                        else:
                            hidden_value = np.append(hidden_value, saida[2].flatten())
                            softmax_value = np.append(softmax_value, saida[3].flatten())
                            
                    
                    
                    
                # train(windowIdxs[idx * self.batchSize : (idx + 1) * self.batchSize], idx, lr)    
            
            print 'Time to training the epoch  ' + str(time.time() - t1)
            
            # Evaluate the model when necessary
            for l in self.listeners:
                l.afterEpoch(epoch)
            
            if debug_mode:
                if (epoch)%5 == 0: 
                    
                    if self.charModel:
                        charhidden_value = np.array(charhidden_value).flatten()
                        charmodel_value = np.array(charmodel_value).flatten()
                    wordmodel_value = np.array(wordmodel_value).flatten()
                    if self.task == 'sentiment_analysis':
                        senthidden_value = np.array(senthidden_value).flatten()
                        sentmodel_value = np.array(sentmodel_value).flatten()
                    hidden_value = np.array(hidden_value).flatten()
                    softmax_value = np.array(softmax_value).flatten()
                    
                    
                    if self.charModel:                
                        self.saveHist(charhidden_value, 20, 'Char Hidden value', 'num', 
                                      model_description + ' - char hidden- until '+str(epoch), [-1, 1, 0, 1200000], 
                                      folder+'charhidden_'+str(epoch/5)+'.png');
                        self.saveHist(charmodel_value, 20, 'Char Conv value', 'num', 
                                      model_description + ' - char conv- until '+str(epoch), [-1, 1, 0, 1200000], 
                                      folder+'charconv_'+str(epoch/5)+'.png');
    
                    self.saveHist(wordmodel_value, 20, 'word embeddings value', 'num', 
                                  model_description + ' - word embedd- until '+str(epoch), [-10, 10, 0, 1000000], 
                                  folder+'word_'+str(epoch/5)+'.png');
                                  
                    if self.task == 'sentiment_analysis':
                        self.saveHist(senthidden_value, 20, 'sentence hidden value', 'num', 
                                      model_description + ' - sent hidden- until '+str(epoch), [-10, 10, 0, 2000000], 
                                      folder+'senthidden_'+str(epoch/5)+'.png');#1-->2
                        self.saveHist(sentmodel_value, 20, 'sentence Conv value', 'num', 
                                      model_description + ' - sent conv- until '+str(epoch), [-10, 10, 0, 100000], 
                                      folder+'sentconv_'+str(epoch/5)+'.png');
                    
                    self.saveHist(hidden_value, 20, 'hidden value', 'num', 
                                  model_description + ' - hidden act - until '+str(epoch), [-1, 1, 0, 200000], 
                                  folder+'hidden_'+str(epoch/5)+'.png');#20-->12 
                    self.saveHist(softmax_value, 20, 'soft prediction', 'num', 
                                  model_description + ' - softmax prediction- until '+str(epoch), [0, 5, 0, 2000], 
                                  folder+'soft_'+str(epoch/5)+'.png');
            
                

    
    def predict(self, inputData):
        raise NotImplementedError();

    def isAdaGrad(self):
        return self.__adaGrad
    
    def saveHist(self, values, bin, xlabel, ylabel, title, axis, filename):
        plt.figure()
        n, bins, patches = plt.hist(values, bins=bin)
        #l = plt.plot(bins, 'r--', linewidth=1)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.axis(axis)
        plt.grid(True)
        
        plt.savefig(filename)

        
    