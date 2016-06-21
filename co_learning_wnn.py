#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import logging
import logging.config
import os
import random
import sys

import numpy as np
import theano.tensor as T

from DataOperation.CoLearningDataReader import CoLearningDataReader
from DataOperation.Embedding import EmbeddingFactory, RandomUnknownStrategy
from DataOperation.InputGenerator.BatchIterator import SyncBatchIterator
from DataOperation.InputGenerator.CoLearningLabelGenerator import CoLearningLabelGenerator
from DataOperation.InputGenerator.WindowGenerator import WindowGenerator
from DataOperation.Lexicon import Lexicon
from DataOperation.TokenDatasetReader import TokenLabelReader
from ModelOperation.CoLearningWnnLoss import CoLearningWnnLoss
from ModelOperation.Model import Model
from ModelOperation.Prediction import ArgmaxPrediction, CoLearningWnnPrediction
from NNet.ActivationLayer import ActivationLayer, softmax, tanh
from NNet.EmbeddingLayer import EmbeddingLayer
from NNet.FlattenLayer import FlattenLayer
from NNet.LinearLayer import LinearLayer
from NNet.WeightGenerator import ZeroWeightGenerator
from Optimizers.Adagrad import Adagrad
from Parameters.JsonArgParser import JsonArgParser

CO_LEARNING_PARAMETERS = {
    "token_label_separator": {"required": True,
                              "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "filters": {"required": True,
                "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},
    "lambda": {"desc": "", "required": True},

    "train_supervised": {"desc": "Supervised Training File Path"},
    "train_unsupervised": {"desc": "Unsupervised Training File Path"},
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
    "word_embedding1": {"desc": "word embedding File Path"},
    "word_embedding2": {"desc": "word embedding File Path"},
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
    "seed": {"desc": ""},
}


def main(**kwargs):
    log = logging.getLogger(__name__)
    log.info(kwargs)

    if kwargs["seed"] != None:
        random.seed(kwargs["seed"])
        np.random.seed(kwargs["seed"])

    filters = []

    for filterName in kwargs["filters"]:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    # Get the inputs and output
    labelLexicon = Lexicon()
    wordWindowSize = kwargs["word_window_size"]
    hiddenLayerSize = kwargs["hidden_size"]
    batchSize = kwargs["batch_size"]
    startSymbol = kwargs["start_symbol"]
    numEpochs = kwargs["num_epochs"]
    lr = kwargs["lr"]

    log.info("Reading training examples")

    embedding1 = EmbeddingFactory().createFromW2V(kwargs["word_embedding1"], RandomUnknownStrategy())
    embedding2 = EmbeddingFactory().createFromW2V(kwargs["word_embedding2"], RandomUnknownStrategy())

    inputGenerator1 = WindowGenerator(wordWindowSize, embedding1, filters, startSymbol)
    inputGenerator2 = WindowGenerator(wordWindowSize, embedding2, filters, startSymbol)

    outputGenerator = CoLearningLabelGenerator(labelLexicon)

    trainDatasetReader = CoLearningDataReader(kwargs["train_supervised"], kwargs["train_unsupervised"],
                                              kwargs["token_label_separator"])

    trainReader = SyncBatchIterator(trainDatasetReader, [inputGenerator1, inputGenerator2], outputGenerator, batchSize)

    embedding1.stopAdd()
    embedding2.stopAdd()
    labelLexicon.stopAdd()

    # Get dev inputs and output
    log.info("Reading development examples")
    devDatasetReader = TokenLabelReader(kwargs["dev"], kwargs["token_label_separator"])
    devReader = SyncBatchIterator(devDatasetReader, [inputGenerator1, inputGenerator2], outputGenerator, batchSize,
                                  shuffle=False)

    ## Learner1
    log.info("Reading W2v File1")

    input1 = T.lmatrix(name="input1")

    embeddingLayer1 = EmbeddingLayer(input1, embedding1.getEmbeddingMatrix(), trainable=False)
    flatten1 = FlattenLayer(embeddingLayer1)

    linear11 = LinearLayer(flatten1, wordWindowSize * embedding1.getEmbeddingSize(), hiddenLayerSize)
    act11 = ActivationLayer(linear11, tanh)

    linear12 = LinearLayer(act11, hiddenLayerSize, labelLexicon.getLen(), weightInitialization=ZeroWeightGenerator())
    act12 = ActivationLayer(linear12, softmax)

    ## Learner2
    log.info("Reading W2v File2")

    input2 = T.lmatrix(name="input2")

    embeddingLayer2 = EmbeddingLayer(input2, embedding2.getEmbeddingMatrix(), trainable=False)
    flatten2 = FlattenLayer(embeddingLayer2)

    linear21 = LinearLayer(flatten2, wordWindowSize * embedding2.getEmbeddingSize(), hiddenLayerSize)
    act21 = ActivationLayer(linear21, tanh)

    linear22 = LinearLayer(act21, hiddenLayerSize, labelLexicon.getLen(), weightInitialization=ZeroWeightGenerator())
    act22 = ActivationLayer(linear22, softmax)

    y = T.lvector("y")

    # Set loss and prediction and retrieve all layers
    output1 = act12.getOutput()
    prediction1 = ArgmaxPrediction(1).predict(output1)

    output2 = act22.getOutput()
    prediction2 = ArgmaxPrediction(1).predict(output2)

    loss = CoLearningWnnLoss(kwargs["lambda"]).calculateError([output1, output2], [prediction1, prediction2], y)
    prediction = CoLearningWnnPrediction().predict([output1, output2])
    allLayers = act12.getLayerSet() | act22.getLayerSet()

    # Creates model
    model = Model([input1, input2], y)

    # Compile Model
    opt = Adagrad(lr=lr, decay=1.0)

    log.info("Compiling the model")
    model.compile(allLayers, opt, prediction, loss, ["acc"])

    # Training Model
    model.train(trainReader, numEpochs, devReader)


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = JsonArgParser(CO_LEARNING_PARAMETERS).parse(sys.argv[1])
    main(**parameters)
