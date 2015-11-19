'''
Created on Nov 19, 2015

@author: eraldo
'''

class Layer(object):
    '''
    Base class for neural network layers. A layer must have an input, some
    (learnable) parameters and an output. The output must be a function of the
    input and the parameters. Optionally, a layer can compute its gradient-based
    updates given a cost function. This is useful for structured layers like
    Word Embeddings (see EmbeddingLayer class) which can compute a much more
    efficient update rule since just a small fraction of its parameters are
    used in each minibatch (or online) iteration.
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
        :return a list comprising all parameters of this layer.
        '''
        raise NotImplementedError()
    
    def getUpdates(self, cost, learningRate):
        '''
        Some layers can have a more efficient way of updating their parameters.
        This is usually true for structured layers (not fully connected), like
        word embeddings.
        
        :return None, if this layer update can be directly computed from the 
            gradient of the cost function w.r.t. its parameters.
            Or, return a list of tuples (var, newValue) that updates this
            layer parameters w.r.t. the given cost function and learning rate.
        '''
        raise NotImplementedError()
