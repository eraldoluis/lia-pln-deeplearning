#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import importlib
import json
import logging
import logging.config
import os
import random
import sys
import time
from time import time

import numpy as np
import theano.tensor as T

from DataOperation.Embedding import EmbeddingFactory, RandomUnknownStrategy, ChosenUnknownStrategy
from DataOperation.InputGenerator.BatchIterator import SyncBatchIterator
from DataOperation.InputGenerator.LabelGenerator import LabelGenerator
from DataOperation.InputGenerator.WindowGenerator import WindowGenerator
from DataOperation.Lexicon import Lexicon, createLexiconUsingFile
from DataOperation.TokenDatasetReader import TokenLabelReader
from ModelOperation.Model import Model, ModelUnit
from ModelOperation.Objective import NegativeLogLikelihood
from ModelOperation.Prediction import ArgmaxPrediction
from ModelOperation.SaveModelCallback import ModelWriter, SaveModelCallback
from NNet.ActivationLayer import ActivationLayer, softmax, tanh, sigmoid
from NNet.EmbeddingLayer import EmbeddingLayer
from NNet.FlattenLayer import FlattenLayer
from NNet.LinearLayer import LinearLayer
from NNet.WeightGenerator import ZeroWeightGenerator, GlorotUniform, SigmoidGlorot
from Optimizers.Adagrad import Adagrad
from Optimizers.SGD import SGD
from Parameters.JsonArgParser import JsonArgParser

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
                              "The possible values are: max_min, mean_normalization or none"},
    "label_file": {"desc": ""},
    "lambda": {"desc": ""}
}


class WNNModelWritter(ModelWriter):
    def __init__(self, savePath, embeddingLayer, linearLayer1, linearLayer2, embedding, lexiconLabel,
                 hiddenActFunction):
        '''
        :param savePath: path where the model will be saved

        :type embeddingLayer: NNet.EmbeddingLayer.EmbeddingLayer
        :type linearLayer1: NNet.LinearLayer.LinearLayer
        :type linearLayer2: NNet.LinearLayer.LinearLayer
        :type embedding: DataOperation.Embedding.Embedding
        '''
        self.__savePath = savePath
        self.__embeddingLayer = embeddingLayer
        self.__linear1 = linearLayer1
        self.__linear2 = linearLayer2
        self.__embedding = embedding
        self.__logging = logging.getLogger(__name__)
        self.__labelLexicon = lexiconLabel
        self.__hiddenActFunction = hiddenActFunction

    def save(self):

        begin = int(time())
        # Saving embedding
        wbFile = codecs.open(self.__savePath + ".wv", "w", encoding="utf-8")
        lexicon = self.__embedding.getLexicon()
        listWords = lexicon.getLexiconList()
        wordEmbeddings = self.__embeddingLayer.getParameters()[0].get_value()

        # wbFile.write(unicode(len(listWords)))
        # wbFile.write(" ")
        # wbFile.write(unicode(self.__embedding.getEmbeddingSize()))
        # wbFile.write("\n")
        #
        # for a in xrange(len(listWords)):
        #     wbFile.write(listWords[a])
        #     wbFile.write(' ')
        #
        #     for i in wordEmbeddings[a]:
        #         wbFile.write(unicode(i))
        #         wbFile.write(' ')
        #
        #     wbFile.write('\n')

        wbFile.close()

        # Savings labels
        param = {
            "labels": self.__labelLexicon.getLexiconList(),
            "hiddenActFunction": self.__hiddenActFunction,
            "unknown": lexicon.getLexicon(lexicon.getUnknownIndex())
        }

        with codecs.open(self.__savePath + ".param", "w", encoding="utf-8") as paramsFile:
            json.dump(param, paramsFile, encoding="utf-8")

        weights = {}

        W1, b1 = self.__linear1.getParameters()
        weights["W_Hidden"] = W1.get_value()
        weights["b_Hidden"] = b1.get_value()

        W2, b2 = self.__linear2.getParameters()

        weights["W_Softmax"] = W2.get_value()
        weights["b_Softmax"] = b2.get_value()

        np.save(self.__savePath, weights)

        self.__logging.info("Model Saved in %d", int(time()) - begin)


def mainWnn(**kwargs):
    log = logging.getLogger(__name__)
    log.info(kwargs)

    if kwargs["seed"] != None:
        random.seed(kwargs["seed"])
        np.random.seed(kwargs["seed"])

    lr = kwargs["lr"]
    startSymbol = kwargs["start_symbol"]
    endSymbol = kwargs["end_symbol"]
    numEpochs = kwargs["num_epochs"]
    shuffle = kwargs["shuffle"]
    normalizeMethod = kwargs["normalization"].lower() if kwargs["normalization"] is not None else None
    wordWindowSize = kwargs["word_window_size"]
    hiddenLayerSize = kwargs["hidden_size"]

    if kwargs["alg"] == "window_stn":
        isSentenceModel = True
    elif kwargs["alg"] == "window_word":
        isSentenceModel = False
    else:
        raise Exception("The value of model_type isn't valid.")

    batchSize = -1 if isSentenceModel else kwargs["batch_size"]
    filters = []

    for filterName in kwargs["filters"]:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    loadPath = kwargs["load_model"]

    if loadPath:
        with codecs.open(loadPath + ".param", "r", encoding="utf-8") as paramsFile:
            param = json.load(paramsFile, encoding="utf-8")

        hiddenActFunctionName = param['hiddenActFunction']
        hiddenActFunction = method_name(hiddenActFunctionName)

        # Loading Embedding
        log.info("Loading Model")
        embedding = EmbeddingFactory().createFromW2V(loadPath + ".wv", ChosenUnknownStrategy(param["unknown"]))
        labelLexicon = Lexicon()

        for l in param["labels"]:
            labelLexicon.put(l)

        labelLexicon.stopAdd()

        # Loading model
        weights = np.load(loadPath + ".npy").item(0)

        W1 = weights["W_Hidden"]
        b1 = weights["b_Hidden"]
        W2 = weights["W_Softmax"]
        b2 = weights["b_Softmax"]

        hiddenLayerSize = b1.shape[0]
    else:
        W1 = None
        b1 = None
        W2 = None
        b2 = None
        hiddenActFunctionName = kwargs["hidden_activation_function"]
        hiddenActFunction = method_name(hiddenActFunctionName)

        if kwargs["word_embedding"]:
            log.info("Reading W2v File")
            embedding = EmbeddingFactory().createFromW2V(kwargs["word_embedding"], RandomUnknownStrategy())
        else:
            embedding = EmbeddingFactory().createRandomEmbedding(kwargs["word_emb_size"])

        # Get the inputs and output
        if kwargs["label_file"]:
            labelLexicon = createLexiconUsingFile(kwargs["label_file"])
        else:
            labelLexicon = Lexicon()

        if kwargs["load_hidden_layer"]:
            # Loading Hidden Layer
            log.info("Loading Hidden Layer")

            mdaWeights = np.load(kwargs["load_hidden_layer"]).item(0)

            W1 = mdaWeights["W_Encoder"]
            b1 = mdaWeights["b_Encoder"]

            hiddenLayerSize = b1.shape[0]

    inputGenerator = WindowGenerator(wordWindowSize, embedding, filters, startSymbol, endSymbol)
    outputGenerator = LabelGenerator(labelLexicon)

    if kwargs["train"]:
        log.info("Reading training examples")

        trainDatasetReader = TokenLabelReader(kwargs["train"], kwargs["token_label_separator"])
        trainReader = SyncBatchIterator(trainDatasetReader, [inputGenerator], outputGenerator, batchSize,
                                        shuffle=shuffle)
        embedding.stopAdd()
        labelLexicon.stopAdd()

        # Get dev inputs and output
        dev = kwargs["dev"]

        if dev:
            log.info("Reading development examples")
            devDatasetReader = TokenLabelReader(kwargs["dev"], kwargs["token_label_separator"])
            devReader = SyncBatchIterator(devDatasetReader, [inputGenerator], outputGenerator, sys.maxint, shuffle=False)
        else:
            devReader = None
    else:
        trainReader = None
        devReader = None

    weightInit = SigmoidGlorot() if hiddenActFunction == sigmoid else GlorotUniform()

    if normalizeMethod == "min_max":
        log.info("Normalization: min max")
        embedding.minMaxNormalization()
    elif normalizeMethod == "mean_normalization":
        log.info("Normalization: mean normalization")
        embedding.meanNormalization()

    if normalizeMethod is not None and loadPath is not None:
        log.warn("The word embedding of model was normalized. This can change the result of test.")

    if isSentenceModel:
        raise NotImplementedError("ModelOperation of sentence window was't implemented yet.")
    else:
        input = T.lmatrix("window_words")

        embeddingLayer = EmbeddingLayer(input, embedding.getEmbeddingMatrix(), trainable=False)
        flatten = FlattenLayer(embeddingLayer)

        linear1 = LinearLayer(flatten, wordWindowSize * embedding.getEmbeddingSize(), hiddenLayerSize, W=W1, b=b1,
                              weightInitialization=weightInit)
        act1 = ActivationLayer(linear1, hiddenActFunction)

        linear2 = LinearLayer(act1, hiddenLayerSize, labelLexicon.getLen(), W=W2, b=b2,
                              weightInitialization=ZeroWeightGenerator())
        act2 = ActivationLayer(linear2, softmax)
        prediction = ArgmaxPrediction(1).predict(act2.getOutput())

    y = T.lvector("y")

    if kwargs["decay"].lower() == "normal":
        decay = 0.0
    elif kwargs["decay"].lower() == "divide_epoch":
        decay = 1.0

    if kwargs["adagrad"]:
        log.info("Using Adagrad")
        opt = Adagrad(lr=lr, decay=decay)
    else:
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)

    # Printing embedding information
    dictionarySize = embedding.getNumberOfEmbeddings()
    embeddingSize = embedding.getEmbeddingSize()
    log.info("Number of dictionary and embedding size: %d and %d" % (dictionarySize, embeddingSize))

    # Compiling
    loss = NegativeLogLikelihood().calculateError(act2.getOutput(), prediction, y)

    if kwargs["lambda"]:
        _lambda = kwargs["lambda"]
        log.info("Using L2 with lambda= %.2f", _lambda)
        loss += _lambda * (T.sum(T.square(linear1.getParameters()[0])))

    wnnModel = Model()

    modelUnit = ModelUnit("wnn", [input], y, loss, prediction=prediction)

    wnnModel.addTrainingModelUnit(modelUnit, ["loss", "acc"])
    wnnModel.setEvaluatedModelUnit(modelUnit, ["loss", "acc"])

    wnnModel.compile([(opt, act2.getLayerSet())])

    # Training
    if trainReader:
        callback = []

        if kwargs["save_model"]:
            savePath = kwargs["save_model"]
            modelWriter = WNNModelWritter(savePath, embeddingLayer, linear1, linear2, embedding, labelLexicon,
                                          hiddenActFunctionName)
            callback.append(SaveModelCallback(modelWriter, "eval_acc", True))

        log.info("Training")
        wnnModel.train([trainReader], numEpochs, devReader, callbacks=callback)

    # Testing
    if kwargs["test"]:
        log.info("Reading test examples")
        testDatasetReader = TokenLabelReader(kwargs["test"], kwargs["token_label_separator"])
        testReader = SyncBatchIterator(testDatasetReader, [inputGenerator], outputGenerator, batchSize, shuffle=False)

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

    parameters = JsonArgParser(WNN_PARAMETERS).parse(sys.argv[1])
    mainWnn(**parameters)
