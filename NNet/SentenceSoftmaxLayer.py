#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as T
import theano
from NNet.Util import WeightBottou88Generator, generateRandomNumberUniformly
from _collections import deque
from NNet.Layer import Layer

class SentenceSoftmaxLayer(Layer):
   
    def __init__(self, _input, lenIn, lenOut):
        # Base class constructor.
        Layer.__init__(self, _input)
        
        self.W = theano.shared(
            value=WeightBottou88Generator().generateWeight(lenIn, lenOut),
            name='W_softmax',
            borrow=True
        )

        self.b = theano.shared(
            value=WeightBottou88Generator().generateWeight(lenOut),
            name='b_softmax',
            borrow=True
        )
        
        self.emissionValues = T.dot(_input, self.W) + self.b
        self.transitionValues = theano.shared(
                                    generateRandomNumberUniformly(-1.0, 1.0, lenOut, lenOut + 1),
                                    name="transitionValues",
                                    borrow=True)
        self.numClasses = lenOut;

        # parameters of the model
        self.params = [self.W, self.b, self.transitionValues]
        
        # Implementação para calcular o valor de todos caminhos
        numWords = self.emissionValues.shape[0]
        
        # Calculando delta 0 
        x = self.transitionValues[:, 0]
        k = T.max(x)
        x = x - k  
         
        delta = self.emissionValues[0] + T.log(T.exp(x)) + k
         
        """
        A primeira execução do scan é totalmente inútil. 
        Foi necessário fazer deste jeito, pois as frases com uma só palavra fazem com que o n_steps do scan fosse igual à 0,
        já que no código anterior n_steps = numWords - 1. Porém, o scan,na versão 0.7, não suporta n_steps igual a zero.
        """
        def stepToCalculateAllPath(posWord, delta):
            transitionValuesWithoutStarting = self.transitionValues[ : , 1:]
             
            x = transitionValuesWithoutStarting.T + delta
            k = T.max(x, axis=1, keepdims=True)
            x = x - k
             
            newDelta = self.emissionValues[posWord] + T.log(T.sum(T.exp(x), axis=1)) + T.flatten(k)
             
            return T.switch(T.eq(posWord, 0), delta, newDelta),

        # Neste scan não é necessário passar o updates para o function,
        #     pois nenhuma shared variable será alterada durante o scan.
        result, _ = theano.scan(fn=stepToCalculateAllPath,
                                sequences=T.arange(0, numWords),
                                outputs_info=delta,
                                n_steps=numWords)

        # Somente o resultado da última iteração é necessário.
        x = result[-1]
        
        # Truque para evitar overflow do exp.
        k = T.max(x)
        x = x - k

        self.sumValueAllPaths = T.log(T.sum(T.exp(x))) + k

        # Implementação do algoritmo de Viterbi.
        delta = self.emissionValues[0] + self.transitionValues[:, 0]
        argMax = T.zeros((numWords - 1, self.numClasses))
        
        def viterbiStep(posWord, delta, argMax):
            transitionValuesWithoutStarting = self.transitionValues[ T.arange(self.numClasses) , 1:]
            max_argmax = T.max_and_argmax(transitionValuesWithoutStarting.T + delta, axis=1);
            delta = self.emissionValues[posWord] + max_argmax[0]
            argMax = T.set_subtensor(argMax[posWord - 1], max_argmax[1])
              
            return [delta, argMax];
  
        [maxi, argMax], updates = theano.scan(fn=viterbiStep ,
            sequences=T.arange(1, numWords),
            outputs_info=[delta, argMax],
            n_steps=numWords - 1)

        lastClass = T.argmax(maxi[-1])

        self.viterbi = theano.function(inputs=[], outputs=[lastClass, argMax[-1]] , updates=updates)
        
        # Calcula o viterbi para uma frase com um word
        argmaxDelta = T.argmax(delta)
        self.viterbiSentenceOneWord = theano.function(inputs=[], outputs=argmaxDelta)
    
    def getSumPathY(self, y):
        # yl guardará a classe anterior a palavra i , enquanto
        # ymod guardará a classe atual da palavra i. É somado mais 1 a yl,
        # pois a coluna 0 da matrix transitionValues se refere aos valores de cada classe estar começando uma frase.    
        ymod = y[:(y.shape[0] - 1)]
        yl = y[1:] + 1
        
        return T.sum(self.emissionValues[T.arange(self.emissionValues.shape[0]), y]) + self.transitionValues[y[0]][0] + T.sum(self.transitionValues[ymod, yl]);
    
    def getLogOfSumAllPathY(self):
        return self.sumValueAllPaths 
    
    def getOutput(self):
        raise  NotImplemented("Esta classe não implementa este método")
    
    def getParameters(self):
        return self.params
    
    def predict(self, numWords):
        sequencia = deque()
        
        if numWords == 1:
            sequencia.appendleft(int(self.viterbiSentenceOneWord()))
        else:
            resultadoViterbi = self.viterbi()
            lastClass = int(resultadoViterbi[0])
            classesByWord = resultadoViterbi[1]
            
            sequencia.appendleft(lastClass)
            
            i = len(classesByWord) - 1
            
            while i > -1:
                lastClass = int(classesByWord[i][lastClass])
                sequencia.appendleft(lastClass)
                i -= 1
        
        return list(sequencia)
    
    def getPrediction(self):
        raise  NotImplemented("Esta classe não implementa este método")
