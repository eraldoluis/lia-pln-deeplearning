import theano.tensor as T
import numpy
import theano
from NNet.Util import defaultGradParameters
import NNet
from test import numWords
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
        # initialize the baises b as a vector of numberClasses 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (numberClasses,),
                dtype=theano.config.floatX
            ),
            name='b_softmax',
            borrow=True
        )
        
        self.emissionValues = T.dot(input.getOutput(), self.W) + self.b
        self.transitionValues = theano.shared(NNet.Util.WeightTanhGenerator().generateWeight(numberClasses, numberClasses + 1))
        self.numberClasses = numberClasses;

        # parameters of the model
        self.params = [self.W, self.b, self.transitionValues]
      
      
    def getSumPathY(self, y):
        # yl guardará a classe anterior a palavra i , enquanto
        # ymod guardará a classe atual da palavra i. É somado mais 1 a yl,
        # pois a coluna 0 da matrix transitionValues se refere aos valores de cada classe estar começando uma frase.    
        ymod = y[:(y.shape[0] - 1)]
        yl = y[1:] + 1
        
        return T.sum(self.emissionValues[T.arange(self.emissionValues.shape[0]), y]) + self.transitionValues[y[0]][0] + T.sum(self.transitionValues[ymod, yl]);
    
    def getLogOfSumAllPathY(self, numWords):
        def stepToCalculateAllPath(posWord,delta):
            transitionValuesWithoutStarting = self.transitionValues[ T.arange(self.numClasses) , 1:]
    
            return self.emissionValues[posWord] + T.log(T.sum(T.exp(transitionValuesWithoutStarting.T + delta),axis=1))
        
        delta = self.emissionValues[0] + T.log(T.exp(self.transitionValues[:,0]))
        
        result , updates = theano.scan(fn=stepToCalculateAllPath ,
            sequences= T.arange(1,numWords),
            outputs_info= delta,
            n_steps = numWords-1)
        
        sumValueAllPaths = T.log(T.sum(T.exp(result[-1])))
         
        return sumValueAllPaths,updates 
    
    def getOutput(self):
        raise  NotImplemented("Esta classe não implementa este método")
    
    def getParameters(self):
        return self.params
    
    
    def predict(self,numWords):
        delta = self.emissionValues[0] + self.transitionValues[:,0]
        argMax = T.zeros((numWords - 1, self.numClasses))
 
        def viterbi(posWord,delta,argMax):
            transitionValuesWithoutStarting = self.transitionValues[ T.arange(self.numClasses) , 1:]
            max_argmax= T.max_and_argmax(transitionValuesWithoutStarting.T + delta,axis=1);
            delta = self.emissionValues[posWord] + max_argmax[0]
            argMax = T.set_subtensor(argMax[posWord - 1], max_argmax[1])
              
            return [delta,argMax];
  
        [max, argMax], updates =  theano.scan(fn= viterbi ,
            sequences= T.arange(1,numWords),
            outputs_info= [delta,argMax],
            n_steps = numWords - 1)

        lastClass = T.argmax(max[-1])
             
        viterbiF = theano.function(inputs=[],outputs=[lastClass,argMax[-1]] , updates=updates)
        
        resultadoViterbi =  viterbiF()
        lastClass = int(resultadoViterbi[0])
        classesByWord = resultadoViterbi[1]
        
        sequencia = deque([lastClass])
        
        i = len(classesByWord) -1
        
        while i > -1:
            lastClass = int(classesByWord[i][lastClass])
            sequencia.appendleft(lastClass)
            i-=1
        
        return tuple(sequencia)
    
    def getPrediction(self):
        return self.y_pred;
    
    def getUpdate(self, cost, learningRate):
        return defaultGradParameters(cost, self.params, learningRate);

