#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import logging
import logging.config
import os
import random
import sys

import numpy as np
import theano
import theano.tensor as T
from theano import printing

from DataOperation.Embedding import EmbeddingFactory, RandomUnknownStrategy
from DataOperation.InputGenerator.BatchIterator import SyncBatchIterator
from DataOperation.InputGenerator.ConstantLabel import ConstantLabel
from DataOperation.InputGenerator.LabelGenerator import LabelGenerator
from DataOperation.InputGenerator.WindowGenerator import WindowGenerator
from DataOperation.Lexicon import createLexiconUsingFile, Lexicon
from DataOperation.TokenDatasetReader import TokenLabelReader, TokenReader
from ModelOperation import ReverseGradientModel
from ModelOperation.Callback import Callback
from ModelOperation.CoLearningModel import CoLearningModel
from ModelOperation.Model import Model, ModelUnit
from ModelOperation.Objective import NegativeLogLikelihood
from ModelOperation.Prediction import ArgmaxPrediction, CoLearningWnnPrediction
from ModelOperation.ReverseGradientModel import ReverseGradientModel
from NNet.ActivationLayer import ActivationLayer, softmax, tanh
from NNet.EmbeddingLayer import EmbeddingLayer
from NNet.FlattenLayer import FlattenLayer
from NNet.LinearLayer import LinearLayer
from NNet.ReverseGradientLayer import ReverseGradientLayer
from NNet.WeightGenerator import ZeroWeightGenerator, GlorotUniform
from Optimizers.Adagrad import Adagrad
from Optimizers.SGD import SGD
from Parameters.JsonArgParser import JsonArgParser
from co_learning_wnn import LossCallback

CO_LEARNING_PARAMETERS = {
    "token_label_separator": {"required": True,
                              "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "filters": {"required": True,
                "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},
    "lambda": {"desc": "", "required": True},
    "label_file": {"desc": "", "required": True},
    "batch_size": {"required": True},

    "train_source": {"desc": "Supervised Training File Path"},
    "train_target": {"desc": "Unsupervised Training File Path"},
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
    "l2": {"default": [None, None]},
}


class ChangeLambda(Callback):
    def __init__(self, lambdaShared, lambdaValue, lossUnsupervisedEpoch):
        self.lambdaShared = lambdaShared
        self.lambdaValue = lambdaValue
        self.lossUnsupervisedEpoch = lossUnsupervisedEpoch

    def onEpochBegin(self, epoch, logs={}):
        e = 1 if epoch >= self.lossUnsupervisedEpoch else 0
        self.lambdaShared.set_value(self.lambdaValue * e)


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
    wordWindowSize = kwargs["word_window_size"]
    hiddenLayerSize = kwargs["hidden_size"]
    batchSize = kwargs["batch_size"]
    startSymbol = kwargs["start_symbol"]
    numEpochs = kwargs["num_epochs"]
    lr = kwargs["lr"]
    tagLexicon = createLexiconUsingFile(kwargs["label_file"])
    _lambda = 0

    domainLexicon = Lexicon()

    domainLexicon.put("0")
    domainLexicon.put("1")
    domainLexicon.stopAdd()

    log.info("Reading training examples")

    log.info("Reading W2v File1")
    embedding1 = EmbeddingFactory().createFromW2V(kwargs["word_embedding"], RandomUnknownStrategy())

    # Source part
    windowSource = T.lmatrix(name="windowSource")
    embeddingLayer1 = EmbeddingLayer(windowSource, embedding1.getEmbeddingMatrix(), trainable=True)
    flatten1 = FlattenLayer(embeddingLayer1)

    linear1 = LinearLayer(flatten1, wordWindowSize * embedding1.getEmbeddingSize(), hiddenLayerSize,
                          weightInitialization=GlorotUniform())
    act1 = ActivationLayer(linear1, tanh)

    supervisedLinear = LinearLayer(act1, hiddenLayerSize, tagLexicon.getLen(),
                                   weightInitialization=ZeroWeightGenerator())
    supervisedSoftmax = ActivationLayer(supervisedLinear, softmax)

    reverseGradientSource = ReverseGradientLayer(act1, _lambda)

    unsupervisedSourceLinear = LinearLayer(reverseGradientSource, hiddenLayerSize, domainLexicon.getLen(),
                                           weightInitialization=ZeroWeightGenerator())
    unsupervisedSourceSoftmax = ActivationLayer(unsupervisedSourceLinear, softmax)

    ## Target Part
    windowTarget = T.lmatrix(name="windowTarget")

    embeddingLayerUnsuper1 = EmbeddingLayer(windowTarget, embeddingLayer1.getParameters()[0], trainable=False)
    flattenUnsuper1 = FlattenLayer(embeddingLayerUnsuper1)

    w, b = linear1.getParameters()
    linearUnsuper1 = LinearLayer(flattenUnsuper1, wordWindowSize * embedding1.getEmbeddingSize(), hiddenLayerSize,
                                 W=w, b=b, trainable=False)
    actUnsupervised1 = ActivationLayer(linearUnsuper1, tanh)

    reverseGradientTarget = ReverseGradientLayer(actUnsupervised1, _lambda)

    w, b = unsupervisedSourceLinear.getParameters()
    unsupervisedTargetLinear = LinearLayer(reverseGradientTarget, hiddenLayerSize, domainLexicon.getLen(), W=w, b=b, trainable=False)
    unsupervisedTargetSoftmax = ActivationLayer(unsupervisedTargetLinear, softmax)

    # Set loss and prediction and retrieve all layers
    supervisedLabel = T.lvector("supervisedLabel")
    unsupervisedLabelSource = T.lvector("unsupervisedLabelSource")
    unsupervisedLabelTarget = T.lvector("unsupervisedLabelTarget")

    supervisedOutput = supervisedSoftmax.getOutput()
    supervisedPrediction = ArgmaxPrediction(1).predict(supervisedOutput)
    supervisedLoss = NegativeLogLikelihood().calculateError(supervisedOutput, supervisedPrediction, supervisedLabel)

    unsupervisedOutputSource = unsupervisedSourceSoftmax.getOutput()
    unsupervisedPredSource = ArgmaxPrediction(1).predict(unsupervisedOutputSource)
    unsupervisedLossSource = NegativeLogLikelihood().calculateError(unsupervisedOutputSource, None,
                                                                    unsupervisedLabelSource)


    unsupervisedOutputTarget = unsupervisedTargetSoftmax.getOutput()
    unsupervisedPredTarget = ArgmaxPrediction(1).predict(unsupervisedOutputTarget)
    unsupervisedLossTarget = NegativeLogLikelihood().calculateError(unsupervisedOutputTarget, None,
                                                                    unsupervisedLabelTarget)

    unsupervisedPrediction = T.concatenate([unsupervisedPredSource,unsupervisedPredTarget])

    loss = supervisedLoss + unsupervisedLossSource + unsupervisedLossTarget

    # Creates model
    model = ReverseGradientModel([windowSource, windowTarget],
                                 [supervisedLabel, unsupervisedLabelSource, unsupervisedLabelTarget])

    opt = SGD(lr=lr, decay=1.0)

    allLayers = supervisedSoftmax.getLayerSet() | unsupervisedSourceSoftmax.getLayerSet() | unsupervisedTargetSoftmax.getLayerSet()

    model.compile(allLayers, opt, supervisedPrediction, unsupervisedPrediction, loss, loss, supervisedLoss, unsupervisedLossSource + unsupervisedLossTarget)

    # Generators
    windowGenerator = WindowGenerator(wordWindowSize, embedding1, filters, startSymbol)
    outputGeneratorTag = LabelGenerator(tagLexicon)
    unsupervisedLabelSource = ConstantLabel(domainLexicon, "0")

    # Reading supervised and unsupervised data sets.
    trainSupervisedDatasetReader = TokenLabelReader(kwargs["train_source"], kwargs["token_label_separator"])
    trainSupervisedBatch = SyncBatchIterator(trainSupervisedDatasetReader, [windowGenerator],
                                             [outputGeneratorTag, unsupervisedLabelSource], batchSize[0])

    # Get Unsupervised Input
    unsupervisedLabelTarget = ConstantLabel(domainLexicon, "1")

    trainUnsupervisedDatasetReader = TokenReader(kwargs["train_target"])
    trainUnsupervisedDatasetBatch = SyncBatchIterator(trainUnsupervisedDatasetReader,
                                                      [windowGenerator],
                                                      [unsupervisedLabelTarget], batchSize[1])

    # Get dev inputs and output
    log.info("Reading development examples")
    devDatasetReader = TokenLabelReader(kwargs["dev"], kwargs["token_label_separator"])
    devReader = SyncBatchIterator(devDatasetReader, [windowGenerator], [outputGeneratorTag], sys.maxint,
                                  shuffle=False)
    # Stopping to add new words
    embedding1.stopAdd()
    tagLexicon.stopAdd()
    domainLexicon.stopAdd()

    # Create Callbacks
    # lambdaChange = ChangeLambda(_lambdaShared, kwargs["lambda"], kwargs["loss_uns_epoch"])

    # Training Model
    model.train([trainSupervisedBatch, trainUnsupervisedDatasetBatch], numEpochs, devReader,
                callbacks=[])


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = JsonArgParser(CO_LEARNING_PARAMETERS).parse(sys.argv[1])
    main(**parameters)

