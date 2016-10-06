#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import importlib
import json
import logging
import logging.config
import numpy as np
import os
import random
import sys
import time
from time import time

import theano.tensor as T
from data.BatchIterator import SyncBatchIterator
from data.CharacterWindowGenerator import CharacterWindowGenerator
from data.WordWindowGenerator import WordWindowGenerator

from args.JsonArgParser import JsonArgParser
from data.Embedding import EmbeddingFactory, RandomUnknownStrategy, ChosenUnknownStrategy
from data.LabelGenerator import LabelGenerator
from data.Lexicon import Lexicon, createLexiconUsingFile
from data.TokenDatasetReader import TokenLabelReader
from model.BasicModel import BasicModel
from model.Objective import NegativeLogLikelihood
from model.Prediction import ArgmaxPrediction
from model.SaveModelCallback import ModelWriter, SaveModelCallback
from nnet.ActivationLayer import ActivationLayer, softmax, tanh, sigmoid
from nnet.ConcatenateLayer import ConcatenateLayer
from nnet.EmbeddingConvolutionalLayer import EmbeddingConvolutionalLayer
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.FlattenLayer import FlattenLayer
from nnet.LinearLayer import LinearLayer
from nnet.WeightGenerator import ZeroWeightGenerator, GlorotUniform, SigmoidGlorot
from optim.Adagrad import Adagrad
from optim.SGD import SGD
from persistence.H5py import H5py
from util.jsontools import dict2obj

WNN_PARAMETERS = {
    "token_label_separator": {"required": True,
                              "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "filters": {"required": True,
                "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},

    "train": {"desc": "Training File Path"},
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},
    "lr": {"desc": "learning rate value"},
    "save_model": {"desc": "Path + basename that will be used to save the weights and embeddings to be saved."},

    "test": {"desc": "Test File Path"},
    "dev": {"desc": "Development File Path"},
    "load_model": {"desc": "Path + basename that will be used to save the weights and embeddings to be loaded."},

    "with_charwnn": {"default": False, "desc": "Enable or disable the charwnn of the model"},
    "conv_size": {"default": 50, "desc": "The number of neurons in the convolutional layer"},
    "char_emb_size": {"default": 10, "desc": "The size of char embedding"},
    "char_window_size": {"default": 5, "desc": "The size of character windows."},
    "charwnn_with_act": {"default": True,
                         "desc": "Enable or disable the use of a activation function in the convolution. "
                                 "When this parameter is true, we use tanh as activation function"},
    "alg": {"default": "window_word",
            "desc": "The type of algorithm to train and test. The posible inputs are: window_word or window_stn"},
    "hidden_size": {"default": 300, "desc": "The number of neurons in the hidden layer"},
    "word_window_size": {"default": 5, "desc": "The size of words for the wordsWindow"},
    "batch_size": {"default": 16},
    "word_emb_size": {"default": 100, "desc": "size of word embedding"},
    "word_embedding": {"desc": "word embedding File Path"},
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc": ""},
    "adagrad": {"desc": "Activate AdaGrad updates.", "default": True},
    "decay": {"default": "DIVIDE_EPOCH",
              "desc": "Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available"},
    "load_hidden_layer": {"desc": "the file which contains weights and bias of pre-trainned hidden layer"},
    "hidden_activation_function": {"default": "tanh",
                                   "desc": "the activation function of the hidden layer. The possible values are: tanh and sigmoid"},
    "shuffle": {"default": True, "desc": "able or disable the shuffle of training examples."},
    "normalization": {"desc": "Choose the normalize method to be applied on  word embeddings. "
                              "The possible values are: minmax or mean"},
    "label_file": {"desc": "file with all possible labels"},
    "lambda_L2": {"desc": "Set the value of L2 coefficient"},
    "charwnn_filters": {"default": [], "desc": "list contains the filters that will be used by charwnn. "
                                               "Each filter is describe by your module name + . + class name"},
}


class WNNModelWritter(ModelWriter):
    def __init__(self, savePath, listOfPersistentObjs):
        """
        :param savePath: path where the model will be save

        :param listOfPersistentObjs: list of PersistentObject. These objects represents the  necessary data to be saved.
        """
        self.__h5py = H5py(savePath)
        self.__listOfPersistentObjs = listOfPersistentObjs

    def save(self):
        begin = int(time())

        for obj in self.__listOfPersistentObjs:
            self.__h5py.save(obj)

        self.__logging.info("Model Saved in %d", int(time()) - begin)


def mainWnn(**kwargs):
    log = logging.getLogger(__name__)
    log.info(kwargs)

    if kwargs.seed:
        random.seed(kwargs.seed)
        np.random.seed(kwargs.seed)

    lr = kwargs.lr
    startSymbol = kwargs.start_symbol
    endSymbol = kwargs.end_symbol
    numEpochs = kwargs.num_epochs
    shuffle = kwargs.shuffle
    normalizeMethod = kwargs.normalization.lower() if kwargs.normalization is not None else None
    wordWindowSize = kwargs.word_window_size
    hiddenLayerSize = kwargs.hidden_size
    hiddenActFunction = kwargs.hidden_activation_function

    withCharWNN = kwargs.with_charwnn
    charEmbeddingSize = kwargs.char_emb_size
    charWindowSize = kwargs.char_window_size

    # TODO: the maximum number of characters of word is fixed in 20.
    numMaxChar = 20

    if kwargs.alg == "window_stn":
        isSentenceModel = True
    elif kwargs.alg == "window_word":
        isSentenceModel = False
    else:
        raise Exception("The value of model_type isn't valid.")

    batchSize = -1 if isSentenceModel else kwargs.batch_size
    filters = []

    log.info("Lendo filtros básicos")

    for filterName in kwargs.filters:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    filterCharwnn = []

    log.info("Lendo filtros do charwnn")

    for filterName in kwargs.charwnn_filters:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filterCharwnn.append(getattr(module_, className)())

    if kwargs.load_model:
        persistentManager = H5py(kwargs.load_model)
        hiddenActFunction = persistentManager["hidden_activation_function"]

    # Creating embeddings
    if kwargs.word_embedding and not kwargs.load_model:
        log.info("Reading W2v File")
        wordEmbedding = EmbeddingFactory().createFromW2V(kwargs.word_embedding, RandomUnknownStrategy(), "word_lexicon")
    else:
        wordEmbedding = EmbeddingFactory().createRandomEmbedding(kwargs.word_emb_size, "word_lexicon")

    if withCharWNN
        startSymbolChar = "</s>"

        # Create the character embedding
        charEmbedding = EmbeddingFactory().createRandomEmbedding(charEmbeddingSize, "char_lexicon")

        # Insert the padding of the character window
        charEmbedding.put(startSymbolChar)

        # Insert the character that will be used to fill the matrix
        # with a dimension lesser than chosen dimension.This enables that the convolution is performed by a matrix multiplication.
        artificialChar = "ART_CHAR"
        charEmbedding.put(artificialChar)

    # Get the inputs and output
    if kwargs.label_file:
        labelLexicon = createLexiconUsingFile(kwargs.label_file)
    else:
        labelLexicon = Lexicon()

    # Build neural network
    if isSentenceModel:
        raise NotImplementedError("Sentence model is not implemented!")
    else:
        wordWindow = T.lmatrix("word_window")
        inputModel = [wordWindow]

        wordEmbeddingLayer = EmbeddingLayer(wordWindow, wordEmbedding.getEmbeddingMatrix(), trainable=True,
                                            name="word_embedding_layer")
        flatten = FlattenLayer(wordEmbeddingLayer)

        if withCharWNN:
            log.info("Using charwnn")
            convSize = kwargs.conv_size

            if kwargs.charwnn_with_act:
                charAct = tanh
            else:
                charAct = None

            charWindowIdxs = T.ltensor4(name="char_window_idx")
            inputModel.append(charWindowIdxs)

            charEmbeddingConvLayer = EmbeddingConvolutionalLayer(charWindowIdxs, charEmbedding.getEmbeddingMatrix(),
                                                                 numMaxChar, convSize, charWindowSize,
                                                                 charEmbeddingSize, charAct,name="char_convolution_layer")
            layerBeforeLinear = ConcatenateLayer([flatten, charEmbeddingConvLayer])
            sizeLayerBeforeLinear = wordWindowSize * (wordEmbedding.getEmbeddingSize() + convSize)
        else:
            layerBeforeLinear = flatten
            sizeLayerBeforeLinear = wordWindowSize * wordEmbedding.getEmbeddingSize()

        hiddenActFunctionName = kwargs.hiddenActFunction
        hiddenActFunction = method_name(hiddenActFunctionName)

        weightInit = SigmoidGlorot() if hiddenActFunction == sigmoid else GlorotUniform()

        linear1 = LinearLayer(layerBeforeLinear, sizeLayerBeforeLinear, hiddenLayerSize,
                              weightInitialization=weightInit,name="linear1")
        act1 = ActivationLayer(linear1, hiddenActFunction)

        linear2 = LinearLayer(act1, hiddenLayerSize, labelLexicon.getLen(), weightInitialization=ZeroWeightGenerator(),name="linear_softmax")
        act2 = ActivationLayer(linear2, softmax)
        prediction = ArgmaxPrediction(1).predict(act2.getOutput())


    if kwargs.load_model:
        objs = list(act2.getLayerSet())
        objs.append(wordEmbedding)

        if withCharWNN:
            objs.append(charEmbedding)

        # Load the model and lexicon
        for o in objs:
            if o.getName():
                persistentManager.load(o)


    inputGenerators = [WordWindowGenerator(wordWindowSize, wordEmbedding, filters, startSymbol, endSymbol)]

    if withCharWNN:
        inputGenerators.append(
            CharacterWindowGenerator(charEmbedding, numMaxChar, charWindowSize, wordWindowSize, artificialChar,
                                     startSymbolChar, startPaddingWrd=startSymbol, endPaddingWrd=endSymbol,
                                     filters=filterCharwnn))

    outputGenerator = LabelGenerator(labelLexicon)

    if kwargs.train:
        log.info("Reading training examples")

        trainDatasetReader = TokenLabelReader(kwargs.train, kwargs.token_label_separator)
        trainReader = SyncBatchIterator(trainDatasetReader, inputGenerators, [outputGenerator], batchSize,
                                        shuffle=shuffle)
        wordEmbedding.stopAdd()

        if withCharWNN:
            charEmbedding.stopAdd()

        labelLexicon.stopAdd()

        # Get dev inputs and output
        dev = kwargs.dev

        if dev:
            log.info("Reading development examples")
            devDatasetReader = TokenLabelReader(kwargs.dev, kwargs.token_label_separator)
            devReader = SyncBatchIterator(devDatasetReader, inputGenerators, [outputGenerator], sys.maxint,
                                          shuffle=False)
        else:
            devReader = None
    else:
        trainReader = None
        devReader = None

    if not normalizeMethod:
        pass
    elif normalizeMethod == "minmax":
        log.info("Normalization: minmax")
        wordEmbedding.minMaxNormalization()
    elif normalizeMethod == "mean":
        log.info("Normalization: mean normalization")
        wordEmbedding.meanNormalization()
    else:
        log.error("Unknown normalization method: %s" % normalizeMethod)
        sys.exit(1)

    if normalizeMethod is not None and loadPath is not None:
        log.warn("The word embedding of model was normalized. This can change the result of test.")

    y = T.lvector("y")

    if kwargs.decay.lower() == "normal":
        decay = 0.0
    elif kwargs.decay.lower() == "divide_epoch":
        decay = 1.0

    if kwargs.adagrad:
        log.info("Using Adagrad")
        opt = Adagrad(lr=lr, decay=decay)
    else:
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)

    # Printing embedding information
    dictionarySize = wordEmbedding.getNumberOfVectors()
    embeddingSize = wordEmbedding.getEmbeddingSize()
    log.info("Size of  word dictionary and word embedding size: %d and %d" % (dictionarySize, embeddingSize))
    if withCharWNN:
        log.info("Size of  char dictionary and char embedding size: %d and %d" % (
            charEmbedding.getNumberOfVectors(), charEmbedding.getEmbeddingSize()))

    # Compiling
    loss = NegativeLogLikelihood().calculateError(act2.getOutput(), prediction, y)

    if kwargs.lambda_L2:
        _lambda = kwargs.lambda_L2
        log.info("Using L2 with lambda= %.2f", _lambda)
        loss += _lambda * (T.sum(T.square(linear1.getParameters()[0])))

    wnnModel = BasicModel(inputModel, [y])

    wnnModel.compile(act2.getLayerSet(), opt, prediction, loss,.loss
    ", "
    acc)

    # Training
    if trainReader:
        callback = []

        if kwargs.save_model:
            savePath = kwargs.save_model
            modelWriter = WNNModelWritter(savePath, wordEmbeddingLayer, linear1, linear2, wordEmbedding, labelLexicon,
                                          hiddenActFunctionName)
            callback.append(SaveModelCallback(modelWriter, "eval_acc", True))

        log.info("Training")
        wnnModel.train(trainReader, numEpochs, devReader, callbacks=callback)

    # Testing
    if kwargs.test:
        log.info("Reading test examples")
        testDatasetReader = TokenLabelReader(kwargs.test, kwargs.token_label_separator)
        testReader = SyncBatchIterator(testDatasetReader, [inputGenerators], outputGenerator, batchSize, shuffle=False)

        log.info("Testing")
        wnnModel.evaluate(testReader, True)


def method_name(hiddenActFunction):
    if hiddenActFunction == "tanh":
        return tanh
    elif hiddenActFunction == "sigmoid":
        return sigmoid
    else:
        raise Exception("'hidden_activation_function' value don't valid.")


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = dict2obj(JsonArgParser(WNN_PARAMETERS).parse(sys.argv[1]))
    mainWnn(**parameters)
