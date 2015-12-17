'''
Created on Nov 19, 2015

@author: eraldo
'''

class Layer(object):
    '''
    Base class for neural network layers. A layer must have an input, some
    (learnable) parameters and an output. The output must be a function of the
    input and the parameters.
    
    Optionally, a layer can compute its gradient-based updates given a cost 
    function. This is useful for structured (not fully connected) layers, like
    Word Embeddings (see EmbeddingLayer class) which can compute gradients and 
    updates in a much more efficient way than simply 

        grad = T.grad(cost, params)
        params = params - learningRate * grad

    since just some of its parameters are used in each learning iteration.
    '''

    def __init__(self, _input):
        '''
        Constructor
        '''
        self.__input = _input

    def getInput(self):
        '''
        :return the symbolic variable representing the input of this layer.
        '''
        return self.__input

    def getOutput(self):
        '''
        :return the symbolic variable representing the output of this layer.
        '''
        raise NotImplementedError()
    
    def getParameters(self):
        '''
        :return a list comprising all parameters (shared variables) of this layer.
        '''
        raise NotImplementedError()
    
    def getStructuredParameters(self):
        '''
        :return a list of parameters (shared variables) whose gradients and 
            updates are computed in a structured way, like in an embedding layer.
        '''
        raise NotImplementedError()
    
    def getDefaultGradParameters(self):
        '''
        :return a list of parameters (shared variables) whose gradients and 
            updates can be computed in the ordinary way, i.e.:
                grad = T.grad(cost, params)
                params = params - learningRate * grad
        '''
        raise NotImplementedError()
    
    def getUpdates(self, cost, learningRate, sumSqGrads=None):
        '''
        Some layers can have an efficient ways of computing the gradient w.r.t. 
        their parameters and the corresponding updates. This is usually the case
        for structured layers (not fully connected), like word embeddings.
        
        :param cost: theano function representing the cost of a training step
        
        :param learningRate: (possibly symbolic) value representing the learning rate.
        
        :param sumSqGrads: (optional) shared variable that store the sum of the
            squared historical gradients, which is used by AdaGrad.
        
        :return [], an empty list, if this layer parameters updates can be 
            directly computed from the gradient of the cost function w.r.t. its
            parameters. Otherwise, return a list of tuples (var, newValue) that 
            updates this layer parameters w.r.t. the given cost function and 
            learning rate.
        '''
        raise NotImplementedError()
