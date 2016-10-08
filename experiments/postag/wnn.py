#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import logging
import logging.config
import os
import random
import sys
import time
from time import time

import numpy as np
import theano.tensor as T

from args.JsonArgParser import JsonArgParser
from data.BatchIterator import SyncBatchIterator
from data.CharacterWindowGenerator import CharacterWindowGenerator
from data.Embedding import Embedding
from data.LabelGenerator import LabelGenerator
from data.Lexicon import Lexicon
from data.TokenDatasetReader import TokenLabelReader
from data.WordWindowGenerator import WordWindowGenerator
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

    "label_file": {"desc": "file with all possible labels"},
    "word_lexicon": {"desc": ""},
    "char_lexicon": {"desc": ""},

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
    "lambda_L2": {"desc": "Set the value of L2 coefficient"},
    "charwnn_filters": {"default": [], "desc": "list contains the filters that will be used by charwnn. "
                                               "Each filter is describe by your module name + . + class name"},

    "create_only_lexicon": {
        "desc": "When this parameter is true, the script creates the lexicons and doesn't train the model."
                "If file exists in the system, so this file won't be overwritten.", "default": False},
}


class WNNModelWritter(ModelWriter):
    def __init__(self, savePath, listOfPersistentObjs, hiddenActFunctionName):
        """
        :param savePath: path where the model will be save

        :param listOfPersistentObjs: list of PersistentObject. These objects represents the  necessary data to be saved.
        """
        self.__h5py = H5py(savePath)
        self.__listOfPersistentObjs = listOfPersistentObjs
        self.__log = logging.getLogger(__name__)

        self.__h5py["hidden_activation_function"] = hiddenActFunctionName

    def save(self):
        begin = int(time())

        for obj in self.__listOfPersistentObjs:
            if obj.getName():
                self.__h5py.save(obj)

        self.__log.info("Model Saved in %d", int(time()) - begin)


def mainWnn(args):
    ################################################
    # Read parameters
    ##############################################

    log = logging.getLogger(__name__)
    log.info(args)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    lr = args.lr
    startSymbol = args.start_symbol
    endSymbol = args.end_symbol
    numEpochs = args.num_epochs
    shuffle = args.shuffle
    normalizeMethod = args.normalization.lower() if args.normalization is not None else None
    wordWindowSize = args.word_window_size
    hiddenLayerSize = args.hidden_size
    hiddenActFunctionName = args.hidden_activation_function
    embeddingSize = args.word_emb_size

    withCharWNN = args.with_charwnn
    charEmbeddingSize = args.char_emb_size
    charWindowSize = args.char_window_size
    startSymbolChar = "</s>"

    # Insert the character that will be used to fill the matrix
    # with a dimension lesser than chosen dimension.This enables that the convolution is performed by a matrix multiplication.
    artificialChar = "ART_CHAR"

    # TODO: the maximum number of characters of word is fixed in 20.
    numMaxChar = 20

    if args.alg == "window_stn":
        isSentenceModel = True
    elif args.alg == "window_word":
        isSentenceModel = False
    else:
        raise Exception("The value of model_type isn't valid.")

    batchSize = -1 if isSentenceModel else args.batch_size
    filters = []

    # Lendo Filtros do wnn
    log.info("Lendo filtros básicos")

    for filterName in args.filters:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    filterCharwnn = []

    # Lendo Filtros do charwnn
    log.info("Lendo filtros do charwnn")

    for filterName in args.charwnn_filters:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filterCharwnn.append(getattr(module_, className)())

    ################################################
    # Create the lexicon and go out after this
    ################################################
    if args.create_only_lexicon:
        inputGenerators = []
        lexiconsToSave = []

        if args.word_lexicon and not os.path.exists(args.word_lexicon):
            wordLexicon = Lexicon("UUUNKKK", "labelLexicon")

            inputGenerators.append(WordWindowGenerator(wordWindowSize, wordLexicon, filters, startSymbol, endSymbol))
            lexiconsToSave.append((wordLexicon, args.word_lexicon))

        if not os.path.exists(args.label_file):
            labelLexicon = Lexicon(None, "labelLexicon")
            outputGenerator = [LabelGenerator(labelLexicon)]
            lexiconsToSave.append((labelLexicon, args.label_file))
        else:
            outputGenerator = None

        if args.char_lexicon and not os.path.exists(args.char_lexicon):
            charLexicon = Lexicon("UUUNKKK", "labelLexicon")

            charLexicon.put(startSymbolChar)
            charLexicon.put(artificialChar)

            inputGenerators.append(
                CharacterWindowGenerator(charLexicon, numMaxChar, charWindowSize, wordWindowSize, artificialChar,
                                         startSymbolChar, startPaddingWrd=startSymbol, endPaddingWrd=endSymbol,
                                         filters=filterCharwnn))

            lexiconsToSave.append((charLexicon, args.char_lexicon))

        if len(inputGenerators) == 0:
            inputGenerators = None

        if not (inputGenerators or outputGenerator):
            log.info("The script didn't generate any lexicon.")
            return

        trainDatasetReader = TokenLabelReader(args.train, args.token_label_separator)
        trainReader = SyncBatchIterator(trainDatasetReader, inputGenerators, outputGenerator, batchSize,
                                        shuffle=shuffle)

        for lexicon, pathToSave in lexiconsToSave:
            lexicon.save(pathToSave)

        log.info("Lexicons were generated with success!")

        return

    ################################################
    # Starting training
    ###########################################
    if args.load_model:
        persistentManager = H5py(args.load_model)
        hiddenActFunctionName = persistentManager["hidden_activation_function"]

    # Read word lexicon and create word embeddings
    if args.load_model:
        wordLexicon = Lexicon.fromPersistentManager(persistentManager, "word_lexicon")
        vectors = EmbeddingLayer.getEmbeddingFromPersistenceManager(persistentManager, "word_embedding_layer")

        wordEmbedding = Embedding(wordLexicon, vectors)

    elif args.word_embedding:
        wordLexicon, wordEmbedding = Embedding.fromWord2Vec(args.word_embedding, "UUUNKKK", "word_lexicon")
    elif args.word_lexicon:
        wordLexicon = Lexicon.fromTextFile(args.word_lexicon, "word_lexicon")
        wordEmbedding = Embedding(wordLexicon, vectors=None, embeddingSize=embeddingSize)
    else:
        log.error("You need to set one of these parameters: load_model, word_embedding or word_lexicon")
        return

    # Read char lexicon and create char embeddings
    if withCharWNN:
        if args.load_model:
            charLexicon = Lexicon.fromPersistentManager(persistentManager, "char_lexicon")
            vectors = EmbeddingConvolutionalLayer.getEmbeddingFromPersistenceManager(persistentManager,
                                                                                     "char_convolution_layer")

            charEmbedding = Embedding(charLexicon, vectors)
        elif args.char_lexicon:
            charLexicon = Lexicon.fromTextFile(args.char_lexicon, "char_lexicon")
            charEmbedding = Embedding(charLexicon, vectors=None, embeddingSize=charEmbeddingSize)
        else:
            log.error("You need to set one of these parameters: load_model or char_lexicon")
            return

    # Read labels
    if args.load_model:
        labelLexicon = Lexicon.fromPersistentManager(persistentManager, "label_lexicon")
    elif args.label_file:
        labelLexicon = Lexicon.fromTextFile(args.label_file, lexiconName="label_lexicon")
    else:
        log.error("You need to set one of these parameters: load_model, word_embedding or word_lexicon")
        return

    # Normalize the word embedding
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

    if normalizeMethod is not None and args.load_model is not None:
        log.warn("The word embedding of model was normalized. This can change the result of test.")

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
            convSize = args.conv_size

            if args.charwnn_with_act:
                charAct = tanh
            else:
                charAct = None

            charWindowIdxs = T.ltensor4(name="char_window_idx")
            inputModel.append(charWindowIdxs)

            charEmbeddingConvLayer = EmbeddingConvolutionalLayer(charWindowIdxs, charEmbedding.getEmbeddingMatrix(),
                                                                 numMaxChar, convSize, charWindowSize,
                                                                 charEmbeddingSize, charAct,
                                                                    name="char_convolution_layer")
            layerBeforeLinear = ConcatenateLayer([flatten, charEmbeddingConvLayer])
            sizeLayerBeforeLinear = wordWindowSize * (wordEmbedding.getEmbeddingSize() + convSize)
        else:
            layerBeforeLinear = flatten
            sizeLayerBeforeLinear = wordWindowSize * wordEmbedding.getEmbeddingSize()

        hiddenActFunction = method_name(hiddenActFunctionName)

        weightInit = SigmoidGlorot() if hiddenActFunction == sigmoid else GlorotUniform()

        linear1 = LinearLayer(layerBeforeLinear, sizeLayerBeforeLinear, hiddenLayerSize,
                              weightInitialization=weightInit, name="linear1")
        act1 = ActivationLayer(linear1, hiddenActFunction)

        linear2 = LinearLayer(act1, hiddenLayerSize, labelLexicon.getLen(), weightInitialization=ZeroWeightGenerator(),
                              name="linear_softmax")
        act2 = ActivationLayer(linear2, softmax)
        prediction = ArgmaxPrediction(1).predict(act2.getOutput())

    if args.load_model:
        # Load the model
        alreadyLoaded = set([wordEmbeddingLayer])

        for o in (act2.getLayerSet() - alreadyLoaded):
            if o.getName():
                persistentManager.load(o)

    inputGenerators = [WordWindowGenerator(wordWindowSize, wordLexicon, filters, startSymbol, endSymbol)]

    if withCharWNN:
        inputGenerators.append(
            CharacterWindowGenerator(charLexicon, numMaxChar, charWindowSize, wordWindowSize, artificialChar,
                                     startSymbolChar, startPaddingWrd=startSymbol, endPaddingWrd=endSymbol,
                                     filters=filterCharwnn))

    outputGenerator = LabelGenerator(labelLexicon)

    if args.train:
        log.info("Reading training examples")

        trainDatasetReader = TokenLabelReader(args.train, args.token_label_separator)
        trainReader = SyncBatchIterator(trainDatasetReader, inputGenerators, [outputGenerator], batchSize,
                                        shuffle=shuffle)

        # Get dev inputs and output
        dev = args.dev

        if dev:
            log.info("Reading development examples")
            devDatasetReader = TokenLabelReader(args.dev, args.token_label_separator)
            devReader = SyncBatchIterator(devDatasetReader, inputGenerators, [outputGenerator], sys.maxint,
                                          shuffle=False)
        else:
            devReader = None
    else:
        trainReader = None
        devReader = None

    y = T.lvector("y")

    if args.decay.lower() == "normal":
        decay = 0.0
    elif args.decay.lower() == "divide_epoch":
        decay = 1.0

    if args.adagrad:
        log.info("Using Adagrad")
        opt = Adagrad(lr=lr, decay=decay)
    else:
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)

    # Printing embedding information
    dictionarySize = wordEmbedding.getNumberOfVectors()

    log.info("Size of  word dictionary and word embedding size: %d and %d" % (dictionarySize, embeddingSize))

    if withCharWNN:
        log.info("Size of  char dictionary and char embedding size: %d and %d" % (
            charEmbedding.getNumberOfVectors(), charEmbedding.getEmbeddingSize()))

    # Compiling
    loss = NegativeLogLikelihood().calculateError(act2.getOutput(), prediction, y)

    if args.lambda_L2:
        _lambda = args.lambda_L2
        log.info("Using L2 with lambda= %.2f", _lambda)
        loss += _lambda * (T.sum(T.square(linear1.getParameters()[0])))

    wnnModel = BasicModel(inputModel, [y])

    wnnModel.compile(act2.getLayerSet(), opt, prediction, loss, ["loss", "acc"])

    # Training
    if trainReader:
        callback = []

        if args.save_model:
            savePath = args.save_model
            objsToSave = list(act2.getLayerSet()) + [wordLexicon, labelLexicon]

            if withCharWNN:
                objsToSave.append(charLexicon)

            modelWriter = WNNModelWritter(savePath, objsToSave, hiddenActFunctionName)
            callback.append(SaveModelCallback(modelWriter, "eval_acc", True))

        log.info("Training")
        wnnModel.train(trainReader, numEpochs, devReader, callbacks=callback)

    # Testing
    if args.test:
        log.info("Reading test examples")
        testDatasetReader = TokenLabelReader(args.test, args.token_label_separator)
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
    mainWnn(parameters)
