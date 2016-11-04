#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano.tensor as T

from nnet.ActivationLayer import tanh, sigmoid, ActivationLayer
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.Layer import Layer
from nnet.LinearLayer import LinearLayer
from nnet.ReshapeLayer import ReshapeLayer
from nnet.WeightGenerator import GlorotUniform, SigmoidGenerator


class EmbeddingConvolutionalLayer(Layer):
    """
    Convolutional layer of embedding features.
    
    The input of this layer is a 4-D tensor whose shape is:
        (numExs, szWrdWin, numMaxCh, szChWin)
    where
        numExs is the number of examples in a training (mini) batch,
        szWrdWin is the size of the word window
            (the convolution is independently performed for each index in this dimension),
        numMaxCh is the number of characters used to represent words
            (the convolution is performed over this dimension),
        szChWin is the size of the character window
            (each input for the convolution filters is composed by this number of features).
    
    The value numMaxCh, the number of characters used to represent a word,
    is fixed for all word to speedup training. For words that are shorter
    than this value, we extend them with an artificial character. For words
    that are longer than this value, we use only the last numMaxCh
    characters in them.
    
    Thus, this layer is not really convolutional (with variable-sized inputs),
    but it is sufficient for many applications and is much faster than an
    ordinary convolutional layer.
    """

    def __init__(self, input, charEmbedding, numMaxCh, convSize, charWindowSize, charEmbSize, charAct=tanh,
                 structGrad=True, trainable=True, name=None):
        """

        :param input: a layer or theano variable

        :param charEmbedding: numpy.array or python list that contains the character vectors

        :param numMaxCh: the number of characters that will be used in a word.
            If the word size is greater than this parameter, so it'll be used the numMaxCh end characters of the word.
            If the word size is lesser than this parameter, so it'll be filled (numMaxCh - word_size) characters at the
                word end.

        :param convSize: number of convolution filters

        :param charWindowSize: the size of the character window

        :param charEmbSize:  the size of the character embedding

        :param charAct: the activation function. If this paramater is None, so no activation function will be used.

        :param structGrad: whether to use structured gradients or not.
            When using small batches (online gradient descent, in the limit),
            the structured gradient is much more efficient because a small
            fraction of word vectors are used on each iteration.
            However, when using large batches (ordinary gradient descent, in the
            limit), ordinary gradients and updates are more efficient because
            most (or all of the) word vectors are used on each iteration.

        :param trainable: set if the layer is trainable or not

        :param name:unique name of the layer. This is use to save the attributes of this object.
        """

        # Input variable for this layer. Its shape is (numExs, szWrdWin, numMaxCh, szChWin)
        # where numExs is the number of examples in the training batch,
        #       szWrdWin is the size of the word window,
        #       numMaxCh is the number of characters used to represent words, and
        #       szChWin is the size of the character window.
        self.input = input
        super(EmbeddingConvolutionalLayer, self).__init__(self.input, trainable, name)

        self.__output = None
        self.__charWindowSize = charWindowSize
        self.__convSize = convSize

        # This is the fixed size of all words.
        self.maxLenWord = numMaxCh

        # Activation function  of hidden layer
        self.charAct = charAct
        self.__structGrad = structGrad

        # We use the symbolic shape of the input to perform all dimension
        # transformations (reshape) necessary for the computation of this layer.
        shape = T.shape(self.input)
        numExs = shape[0]
        szWrdWin = shape[1]
        numMaxCh = shape[2]
        szChWin = shape[3]

        # Character embedding layer.
        self.__embedLayer = EmbeddingLayer(self.input.flatten(2), charEmbedding, structGrad=structGrad,
                                           trainable=self.isTrainable())

        # It chooses, based in the activation function, the way that the weights of liner layer will be initialized.
        if charAct is tanh:
            weightInitialization = GlorotUniform()
        elif charAct is sigmoid:
            weightInitialization = SigmoidGenerator()
        elif charAct is None:
            pass
        else:
            raise Exception("Activation function is not supported")

        # This is the bank of filters. It is an ordinary hidden layer.
        hidInput = ReshapeLayer(self.__embedLayer, (numExs * szWrdWin * numMaxCh, szChWin * charEmbSize))

        self.__linearLayer = LinearLayer(hidInput, charWindowSize * charEmbSize, self.__convSize,
                                         weightInitialization=weightInitialization, trainable=self.isTrainable())

        if charAct:
            self.actLayer = ActivationLayer(self.__linearLayer, self.charAct)
            layerBeforePolling = self.actLayer
        else:
            layerBeforePolling = self.__linearLayer

        # 3-D tensor with shape (numExs * szWrdWin, numMaxCh, convSize).
        # This tensor is used to perform the max pooling along its 2nd dimension.
        o = ReshapeLayer(layerBeforePolling, (numExs * szWrdWin, numMaxCh, convSize))

        # Max pooling layer. Perform a max op along the character dimension.
        # The shape of the output is equal to (numExs*szWrdWin, convSize).
        m = T.max(o.getOutput(), axis=1)

        # The output is a 2-D tensor with shape (numExs, szWrdWin * convSize).
        self.__output = m.reshape((numExs, szWrdWin * convSize))

    def updateAllCharIndexes(self, charIdxWord):
        for Idx in xrange(len(charIdxWord)):
            indexes = []

            for charIdx in xrange(len(charIdxWord[Idx])):
                indexes.append(self.getCharIndexes(charIdx, charIdxWord[Idx]))

            self.AllCharWindowIndexes.append(indexes)

    def getParameters(self):
        return self.__embedLayer.getParameters() + self.__linearLayer.getParameters()

    def getDefaultGradParameters(self):
        return self.__linearLayer.getDefaultGradParameters() + self.__embedLayer.getDefaultGradParameters()

    def getStructuredParameters(self):
        return self.__linearLayer.getStructuredParameters() + self.__embedLayer.getStructuredParameters()

    def getUpdates(self, cost, lr, sumSqGrads=None):
        return self.__embedLayer.getUpdates(cost, lr, sumSqGrads)

    def getNormalizationUpdates(self, strategy, coef):
        return self.__embedLayer.getNormalizationUpdate(strategy, coef)

    def getOutput(self):
        return self.__output

    def getAttributes(self):
        keyValueList = self.__embedLayer.getAttributes().items()
        keyValueList += self.__linearLayer.getAttributes().items()

        dict = {}

        for key,value in keyValueList:
            dict[key] = value

        return dict

    def load(self, attributes):
        self.__embedLayer.load(attributes)
        self.__linearLayer.load(attributes)

    @staticmethod
    def getEmbeddingFromPersistenceManager(persistenceManager, name):
        """
        Return the embedding vector from the database

        :type persistenceManager: persistence.PersistentManager.PersistentManager
        :param persistenceManager:

        :param name: name of object which the embedding was saved as attribute

        :return:
        """
        return EmbeddingLayer.getEmbeddingFromPersistenceManager(persistenceManager,name)

