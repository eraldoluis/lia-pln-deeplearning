"""
Created on Nov 19, 2015

@author: eraldo
"""
import logging
from abc import ABCMeta, abstractmethod

from persistence import PersistentObject


class Layer(PersistentObject):
    """
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


    The methods getAttibutes() and load() of the interface PersistentObject are implemented by Layer.
    However, these methods don't do nothing and it isn't possible to save a subclasse Layer that doesn't overwrite them.
    Besides that, it's important give a unique name to the layer with want to save it.
    """
    __metaclass__ = ABCMeta

    def __init__(self, _input, trainable=True, name=None):
        """
        Constructor
        """
        if isinstance(_input, list):
            self.__input = []
            self.__set = set()

            for i in _input:
                self.__input.append(self.__auxConst(i))
        else:
            self.__set = set()
            self.__input = self.__auxConst(_input)

        self.__set.add(self)
        self.__trainable = trainable

        self.log = logging.getLogger(__name__)

        if name is None:
            self.log.info("The layer object doesn't have a name. It won't be possible to save this object.")

        self._name = name

    def isTrainable(self):
        return self.__trainable

    def __auxConst(self, _input):
        if isinstance(_input, Layer):
            for s in _input.getLayerSet():
                self.__set.add(s)
            return _input.getOutput()

        return _input

    def getLayerSet(self):
        """
        :return returns a set with previous layers and this layer.
        """
        return self.__set

    def getInput(self):
        """
        :return the symbolic variable representing the input of this layer.
        """
        return self.__input

    @abstractmethod
    def getOutput(self):
        """
        :return the symbolic variable representing the output of this layer.
        """
        raise NotImplementedError()

    @abstractmethod
    def getParameters(self):
        """
        :return a list comprising all parameters (shared variables) of this layer.
        """
        raise NotImplementedError()

    @abstractmethod
    def getStructuredParameters(self):
        """
        :return a list of parameters (shared variables) whose gradients and 
            updates are computed in a structured way, like in an embedding layer.
        """
        raise NotImplementedError()

    @abstractmethod
    def getDefaultGradParameters(self):
        """
        :return a list of parameters (shared variables) whose gradients and 
            updates can be computed in the ordinary way, i.e.:
                grad = T.grad(cost, params)
                params = params - learningRate * grad
        """
        raise NotImplementedError()

    @abstractmethod
    def getUpdates(self, cost, learningRate, sumSqGrads=None):
        """
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
        """
        raise NotImplementedError()

    def getName(self):
        return self.__name

    def getAttibutes(self):
        return None

    def load(self,attributes):
        pass
