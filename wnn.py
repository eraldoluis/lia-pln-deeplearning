#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import logging
import os
import sys

from keras.optimizers import SGD

import Model
from DataOperation.Embedding import EmbeddingFactory
from DataOperation.InputGenerator.LabelGenerator import LabelGenerator
from DataOperation.InputGenerator.WindowGenerator import WindowGenerator
from DataOperation.InputGenerator.BatchIterator import SyncBatchIterator
from DataOperation.Lexicon import Lexicon
from DataOperation.TokenDatasetReader import TokenLabelReader
from Model import Model
from Model.Objective import NegativeLogLikelihood
from Model.Prediction import ArgmaxPrediction
from NNet import InputLayer, LinearLayer
from NNet.ActivationLayer import ActivationLayer, softmax, tanh
from NNet.FlattenLayer import FlattenLayer
from NNet.InputLayer import InputLayer
from NNet.LinearLayer import LinearLayer
from NNet.Embedding import EmbeddingLayer
import logging.config

from Optimizers.SGD import SGD
from Parameters.JsonArgParser import JsonArgParser

WNN_PARAMETERS = u'''
{
    "train": {"desc": "Training File Path", "required": true},
    "num_epochs": {"required": "true", "desc": "Number of epochs: how many iterations over the training set." },
    "token_label_separator": { "required": true, "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "lr": {"desc":"learning rate value", "required": true},
    "filters": {"required":true, "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},

    "test": {"desc": "Test File Path"},
    "dev": {"desc": "Development File Path"},
    "alg": {"default":"window_word", "desc": "The type of algorithm to train and test. The posible inputs are: window_word or window_stn"},
    "hidden_size": {"default": 300, "desc": "The number of neurons in the hidden layer"},
    "word_window_size": {"default": 5 , "desc": "The size of words for the wordsWindow" },
    "batch_size": {"default": 16},
    "word_emb_size": {"default": 100, "desc": "size of word embedding"},
    "word_embedding": {"desc": "word embedding File Path"},
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc":""},
    "adagrad": {"desc": "Activate AdaGrad updates.", "default": true},
    "decay": {"default": "normal", "desc": "Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available"}

}
'''


def mainWnn(**kwargs):
    log = logging.getLogger(__name__)
    log.info(kwargs)

    lr = kwargs["lr"]
    wordWindowSize = kwargs["word_window_size"]
    startSymbol = kwargs["start_symbol"]
    endSymbol = kwargs["end_symbol"]
    numEpochs = kwargs["num_epochs"]

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

    if kwargs["word_embedding"]:
        log.info("Reading W2v File")
        embedding = EmbeddingFactory().createFromW2V(kwargs["word_embedding"])
    else:
        embedding = EmbeddingFactory().createEmptyEmbedding(kwargs["word_emb_size"])

    # Get the inputs and output
    labelLexicon = Lexicon()

    log.info("Reading training examples")

    trainDatasetReader = TokenLabelReader(kwargs["train"], kwargs["token_label_separator"])
    inputGenerator = WindowGenerator(wordWindowSize, embedding, filters,
                                     startSymbol, endSymbol)
    outputGenerator = LabelGenerator(labelLexicon)



    trainReader = SyncBatchIterator(trainDatasetReader, [inputGenerator], outputGenerator, batchSize)
    embedding.stopAdd()
    labelLexicon.stopAdd()

    # log.info("Using %d examples from train data set" % (len(trainExamples[0])))

    # Get dev inputs and output
    dev = kwargs["dev"]

    if dev:
        log.info("Reading development examples")
        devDatasetReader = TokenLabelReader(kwargs["dev"], kwargs["token_label_separator"])
        devReader = SyncBatchIterator(devDatasetReader, [inputGenerator], outputGenerator, batchSize)
        # log.info("Using %d examples from development data set" % (len(devExamples[0])))
    else:
        devReader = None

    wordWindowSize = kwargs["word_window_size"]
    hiddenLayerSize = kwargs["hidden_size"]

    if isSentenceModel:
        raise NotImplementedError("Model of sentence window was't implemented yet.")
    else:
        inputLayer = InputLayer(1, name="word_window", dtype='int64')

        embeddingLayer = EmbeddingLayer(inputLayer, embedding.getEmbeddingMatrix())
        flatten = FlattenLayer(embeddingLayer)

        linear1 = LinearLayer(flatten, wordWindowSize * embedding.getEmbeddingSize(), hiddenLayerSize)
        act1 = ActivationLayer(linear1, tanh)

        linear2 = LinearLayer(act1, hiddenLayerSize, labelLexicon.getLen())
        act2 = ActivationLayer(linear2, softmax)

    wnnModel = Model.Model(act2, dimOutput=0, dtype="int64")

    wnnModel.compile(SGD(lr=lr), NegativeLogLikelihood(), ArgmaxPrediction(1), ["acc"])

    wnnModel.train(trainReader, numEpochs, devReader)


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = JsonArgParser(WNN_PARAMETERS).parse(sys.argv[1])
    mainWnn(**parameters)
