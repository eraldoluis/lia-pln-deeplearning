#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T
import numpy
import theano
from NNet.Util import defaultGradParameters
import NNet
from _collections import deque

class SentenceSoftmaxLayer(object):
   
    def __init__(self, input, numberNeuronsPreviousLayer, numberClasses):        
        self.W = theano.shared(
            value=numpy.zeros(
                (numberNeuronsPreviousLayer, numberClasses),
                dtype=theano.config.floatX
            ),
            name='W_softmax',
            borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros(
                (numberClasses,),
                dtype=theano.config.floatX
            ),
            name='b_softmax',
            borrow=True
        )
        
        self.emissionValues = T.dot(input, self.W) + self.b
        self.transitionValues = theano.shared(NNet.Util.WeightTanhGenerator().generateWeight(numberClasses, numberClasses + 1),name="transitionValues")
        self.numClasses = numberClasses;

        # parameters of the model
        self.params = [self.W, self.b, self.transitionValues]
        
        # Implementação para calcular o valor de todos caminhos
        numWords = self.emissionValues.shape[0]
        
        def stepToCalculateAllPath(posWord,delta):
            transitionValuesWithoutStarting = self.transitionValues[ T.arange(self.numClasses) , 1:]
    
            return self.emissionValues[posWord] + T.log(T.sum(T.exp(transitionValuesWithoutStarting.T + delta),axis=1))
        
        delta = self.emissionValues[0] + T.log(T.exp(self.transitionValues[:,0]))
        
        result , updates = theano.scan(fn=stepToCalculateAllPath ,
            sequences= T.arange(1,numWords),
            outputs_info= delta,
            n_steps = numWords-1)
        
        self.sumValueAllPaths = T.log(T.sum(T.exp(result[-1])))

#       Neste scan não é necessário passar o updates para o function
#       , pois nenhuma shared variable será alterada durante o scan
#       
#        self.updatesSumValue = updates
        
        
        # Implementação do viterbi
        delta = self.emissionValues[0] + self.transitionValues[:,0]
        argMax = T.zeros((numWords - 1, self.numClasses))
        
        def viterbiStep(posWord,delta,argMax):
            transitionValuesWithoutStarting = self.transitionValues[ T.arange(self.numClasses) , 1:]
            max_argmax= T.max_and_argmax(transitionValuesWithoutStarting.T + delta,axis=1);
            delta = self.emissionValues[posWord] + max_argmax[0]
            argMax = T.set_subtensor(argMax[posWord - 1], max_argmax[1])
              
            return [delta,argMax];
  
        [max, argMax], updates =  theano.scan(fn= viterbiStep ,
            sequences= T.arange(1,numWords),
            outputs_info= [delta,argMax],
            n_steps = numWords - 1)

        lastClass = T.argmax(max[-1])
             
        self.viterbi = theano.function(inputs=[],outputs=[lastClass,argMax[-1]] , updates=updates)
        
        # Calcula o viterbi para uma frase com um word
        argmaxDelta = T.argmax(delta)
        self.viterbiSentenceOneWord = theano.function(inputs=[],outputs = argmaxDelta)
    
    def getSumPathY(self, y):
        # yl guardará a classe anterior a palavra i , enquanto
        # ymod guardará a classe atual da palavra i. É somado mais 1 a yl,
        # pois a coluna 0 da matrix transitionValues se refere aos valores de cada classe estar começando uma frase.    
        ymod = y[:(y.shape[0] - 1)]
        yl = y[1:] + 1
        
        return T.sum(self.emissionValues[T.arange(self.emissionValues.shape[0]), y]) + self.transitionValues[y[0]][0] + T.sum(self.transitionValues[ymod, yl]);
    
    def getLogOfSumAllPathY(self, numWords):
        return self.sumValueAllPaths 
    
    def getOutput(self):
        raise  NotImplemented("Esta classe não implementa este método")
    
    def getParameters(self):
        return self.params
    
    def predict(self,numWords):
        sequencia = deque()
        
        if numWords == 1:
            sequencia.appendleft(int(self.viterbiSentenceOneWord()))
        else:
            resultadoViterbi =  self.viterbi()
            lastClass = int(resultadoViterbi[0])
            classesByWord = resultadoViterbi[1]
            
            sequencia.appendleft(lastClass)
            
            i = len(classesByWord) -1
            
            while i > -1:
                lastClass = int(classesByWord[i][lastClass])
                sequencia.appendleft(lastClass)
                i-=1
        
        return tuple(sequencia)
    
    def getPrediction(self):
        raise  NotImplemented("Esta classe não implementa este método")
    
    def getUpdate(self, cost, learningRate):
        up = defaultGradParameters(cost, self.params, learningRate)
        
#        up.append(self.updatesSumValue)
        
        return up;
    
    

