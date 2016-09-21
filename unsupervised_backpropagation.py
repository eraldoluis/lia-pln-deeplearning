#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

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

from DataOperation.Embedding import EmbeddingFactory, RandomUnknownStrategy
from DataOperation.InputGenerator.BatchIterator import SyncBatchList
from DataOperation.InputGenerator.ConstantLabel import ConstantLabel
from DataOperation.InputGenerator.LabelGenerator import LabelGenerator
from DataOperation.InputGenerator.WindowGenerator import WindowGenerator
from DataOperation.Lexicon import createLexiconUsingFile, Lexicon
from DataOperation.TokenDatasetReader import TokenLabelReader, TokenReader
from ModelOperation.Callback import Callback
from ModelOperation.Objective import NegativeLogLikelihood
from ModelOperation.Prediction import ArgmaxPrediction
from ModelOperation.GradientReversalModel import GradientReversalModel
from NNet.ActivationLayer import ActivationLayer, softmax, tanh, sigmoid
from NNet.EmbeddingLayer import EmbeddingLayer
from NNet.FlattenLayer import FlattenLayer
from NNet.LinearLayer import LinearLayer
from NNet.GradientReversalLayer import GradientReversalLayer
from NNet.WeightGenerator import ZeroWeightGenerator, GlorotUniform, SigmoidGenerator
from Optimizers import Adagrad
from Optimizers.Adagrad import Adagrad
from Optimizers.SGD import SGD
from Parameters.JsonArgParser import JsonArgParser

UNSUPERVISED_BACKPROPAGATION_PARAMETERS = {
    "token_label_separator": {"required": True,
                              "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "filters": {"required": True,
                "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},
    "label_file": {"desc": "", "required": True},
    "batch_size": {"required": True},
    "alpha": {"desc": "", "required": True},

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
    "seed": {"desc": ""},
    "hidden_size_unsupervised_part": {"default": 0},
    "hidden_size_supervised_part": {"default": 0, "desc": "Set the size of the hidden layer before "
                                                          "the softmax of the supervised part. If the value is 0, "
                                                          "so this hidden isn't put in the NN."},
    "normalization": {"desc": "options = none, zscore, minmax e mean"},
    "additional_dev": {'desc': ""},
    "activation_hidden_extractor": {"default": "tanh", "desc": "This parameter chooses the type of activation function"
                                                               " that will be used in the hidden layer of the extractor. Options: sigmoid or tanh"},
}


class ChangeLambda(Callback):
    def __init__(self, lambdaShared, alpha, maxNumEpoch, height=1, lowerBound=0):
        self.lambdaShared = lambdaShared
        self.alpha = alpha
        self.height = height
        self.lowerBound = lowerBound
        self.maxNumEpoch = float(maxNumEpoch)

    def onEpochBegin(self, epoch, logs={}):
        progress = min(1., epoch / self.maxNumEpoch)
        _lambda = 2. * self.height / (1. + math.exp(-self.alpha * progress)) - self.height + self.lowerBound;

        self.lambdaShared.set_value(_lambda)


class AdditionalDevDataset(Callback):
    def __init__(self, model, sourceDataset, tokenLabelSeparator, windowGenerator, outputGeneratorTag):
        # Get dev inputs and output
        self.log = logging.getLogger(__name__)
        self.log.info("Reading additional dev examples")
        devDatasetReader = TokenLabelReader(sourceDataset, tokenLabelSeparator)
        devReader = SyncBatchList(devDatasetReader, [windowGenerator], [outputGeneratorTag], sys.maxint,
                                  shuffle=False)

        self.devReader = devReader
        self.model = model

    def onEpochEnd(self, epoch, logs={}):
        result = self.model.evaluate(self.devReader, False)
        self.log.info("Additional Dataset: " + str(result["acc"]))


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
    _lambda = theano.shared(0., "lambda")
    useAdagrad = kwargs["adagrad"]
    shuffle = kwargs["shuffle"]
    supHiddenLayerSize = kwargs["hidden_size_supervised_part"]
    unsupHiddenLayerSize = kwargs["hidden_size_unsupervised_part"]
    normalization = kwargs["normalization"]
    activationHiddenExtractor = kwargs["activation_hidden_extractor"]

    if kwargs["decay"].lower() == "normal":
        decay = 0.0
    elif kwargs["decay"].lower() == "divide_epoch":
        decay = 1.0

    # Add the lexicon of target
    domainLexicon = Lexicon()

    domainLexicon.put("0")
    domainLexicon.put("1")
    domainLexicon.stopAdd()

    log.info("Reading training examples")

    log.info("Reading W2v File1")
    embedding1 = EmbeddingFactory().createFromW2V(kwargs["word_embedding"], RandomUnknownStrategy())

    if normalization == "zscore":
        embedding1.zscoreNormalization()
    elif normalization == "minmax":
        embedding1.minMaxNormalization()
    elif normalization == "mean":
        embedding1.meanNormalization()
    elif normalization == "none" or not normalization:
        pass
    else:
        raise Exception()

    # Source part
    windowSource = T.lmatrix(name="windowSource")

    embeddingLayer1 = EmbeddingLayer(windowSource, embedding1.getEmbeddingMatrix(), trainable=True)
    flatten1 = FlattenLayer(embeddingLayer1)

    if activationHiddenExtractor == "tanh":
        log.info("Using tanh in the hidden layer of extractor")

        linear1 = LinearLayer(flatten1, wordWindowSize * embedding1.getEmbeddingSize(), hiddenLayerSize,
                              weightInitialization=GlorotUniform())
        act1 = ActivationLayer(linear1, tanh)
    elif activationHiddenExtractor == "sigmoid":
        log.info("Using sigmoid in the hidden layer of extractor")

        linear1 = LinearLayer(flatten1, wordWindowSize * embedding1.getEmbeddingSize(), hiddenLayerSize,
                              weightInitialization=SigmoidGenerator())
        act1 = ActivationLayer(linear1, sigmoid)
    else:
        raise Exception()

    if supHiddenLayerSize == 0:
        layerBeforeSupSoftmax = act1
        layerSizeBeforeSupSoftmax = hiddenLayerSize
        log.info("It didn't insert the layer before the supervised softmax.")
    else:
        linear2 = LinearLayer(act1, hiddenLayerSize, supHiddenLayerSize,
                              weightInitialization=GlorotUniform())
        act2 = ActivationLayer(linear2, tanh)

        layerBeforeSupSoftmax = act2
        layerSizeBeforeSupSoftmax = supHiddenLayerSize

        log.info("It inserted the layer before the supervised softmax.")

    supervisedLinear = LinearLayer(layerBeforeSupSoftmax, layerSizeBeforeSupSoftmax, tagLexicon.getLen(),
                                   weightInitialization=ZeroWeightGenerator())
    supervisedSoftmax = ActivationLayer(supervisedLinear, softmax)

    gradientReversalSource = GradientReversalLayer(act1, _lambda)

    if unsupHiddenLayerSize == 0:
        layerBeforeUnsupSoftmax = gradientReversalSource
        layerSizeBeforeUnsupSoftmax = hiddenLayerSize
        log.info("It didn't insert the layer before the unsupervised softmax.")
    else:
        unsupervisedSourceLinearBf = LinearLayer(gradientReversalSource, hiddenLayerSize, unsupHiddenLayerSize,
                                                 weightInitialization=GlorotUniform())
        actUnsupervisedSourceBf = ActivationLayer(unsupervisedSourceLinearBf, tanh)

        layerBeforeUnsupSoftmax = actUnsupervisedSourceBf
        layerSizeBeforeUnsupSoftmax = unsupHiddenLayerSize

        log.info("It inserted the layer before the unsupervised softmax.")

    unsupervisedSourceLinear = LinearLayer(layerBeforeUnsupSoftmax, layerSizeBeforeUnsupSoftmax, domainLexicon.getLen(),
                                           weightInitialization=ZeroWeightGenerator())
    unsupervisedSourceSoftmax = ActivationLayer(unsupervisedSourceLinear, softmax)

    ## Target Part
    windowTarget = T.lmatrix(name="windowTarget")

    embeddingLayerUnsuper1 = EmbeddingLayer(windowTarget, embeddingLayer1.getParameters()[0], trainable=True)
    flattenUnsuper1 = FlattenLayer(embeddingLayerUnsuper1)

    w, b = linear1.getParameters()
    linearUnsuper1 = LinearLayer(flattenUnsuper1, wordWindowSize * embedding1.getEmbeddingSize(), hiddenLayerSize,
                                 W=w, b=b, trainable=True)

    if activationHiddenExtractor == "tanh":
        log.info("Using tanh in the hidden layer of extractor")
        actUnsupervised1 = ActivationLayer(linearUnsuper1, tanh)
    elif activationHiddenExtractor == "sigmoid":
        log.info("Using sigmoid in the hidden layer of extractor")
        actUnsupervised1 = ActivationLayer(linearUnsuper1, sigmoid)
    else:
        raise Exception()

    grandientReversalTarget = GradientReversalLayer(actUnsupervised1, _lambda)

    if unsupHiddenLayerSize == 0:
        layerBeforeUnsupSoftmax = grandientReversalTarget
        layerSizeBeforeUnsupSoftmax = hiddenLayerSize
        log.info("It didn't insert the layer before the unsupervised softmax.")
    else:
        w, b = unsupervisedSourceLinearBf.getParameters()
        unsupervisedTargetLinearBf = LinearLayer(grandientReversalTarget, hiddenLayerSize, unsupHiddenLayerSize, W=w,
                                                 b=b, trainable=True)
        actUnsupervisedTargetLinearBf = ActivationLayer(unsupervisedTargetLinearBf, tanh)

        layerBeforeUnsupSoftmax = actUnsupervisedTargetLinearBf
        layerSizeBeforeUnsupSoftmax = unsupHiddenLayerSize

        log.info("It inserted the layer before the unsupervised softmax.")

    w, b = unsupervisedSourceLinear.getParameters()
    unsupervisedTargetLinear = LinearLayer(layerBeforeUnsupSoftmax, layerSizeBeforeUnsupSoftmax, domainLexicon.getLen(),
                                           W=w, b=b,
                                           trainable=True)
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

    # Creates model
    model = GradientReversalModel(windowSource, windowTarget, supervisedLabel, unsupervisedLabelSource,
                                  unsupervisedLabelTarget)

    if useAdagrad:
        log.info("Using ADAGRAD")
        opt = Adagrad(lr=lr, decay=decay)
    else:
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)

    allLayersSource = supervisedSoftmax.getLayerSet() | unsupervisedSourceSoftmax.getLayerSet()
    allLayersTarget = unsupervisedTargetSoftmax.getLayerSet()

    model.compile(allLayersSource, allLayersTarget, opt, supervisedPrediction, unsupervisedPredSource,
                  unsupervisedPredTarget, supervisedLoss, unsupervisedLossSource, unsupervisedLossTarget)

    # Generators
    windowGenerator = WindowGenerator(wordWindowSize, embedding1, filters, startSymbol)
    outputGeneratorTag = LabelGenerator(tagLexicon)
    unsupervisedLabelSource = ConstantLabel(domainLexicon, "0")

    # Reading supervised and unsupervised data sets.
    trainSupervisedDatasetReader = TokenLabelReader(kwargs["train_source"], kwargs["token_label_separator"])
    trainSupervisedBatch = SyncBatchList(trainSupervisedDatasetReader, [windowGenerator],
                                         [outputGeneratorTag, unsupervisedLabelSource], batchSize[0],
                                         shuffle=shuffle)

    # Get Unsupervised Input
    unsupervisedLabelTarget = ConstantLabel(domainLexicon, "1")

    trainUnsupervisedDatasetReader = TokenReader(kwargs["train_target"])
    trainUnsupervisedDatasetBatch = SyncBatchList(trainUnsupervisedDatasetReader,
                                                  [windowGenerator],
                                                  [unsupervisedLabelTarget], batchSize[1], shuffle=shuffle)

    # Get dev inputs and output
    log.info("Reading development examples")
    devDatasetReader = TokenLabelReader(kwargs["dev"], kwargs["token_label_separator"])
    devReader = SyncBatchList(devDatasetReader, [windowGenerator], [outputGeneratorTag], sys.maxint,
                              shuffle=False)
    # Stopping to add new words
    embedding1.stopAdd()
    tagLexicon.stopAdd()
    domainLexicon.stopAdd()

    callbacks = []

    callbacks.append(ChangeLambda(_lambda, kwargs["alpha"], numEpochs))

    if kwargs["additional_dev"]:
        callbacks.append(
            AdditionalDevDataset(model, kwargs["additional_dev"], kwargs["token_label_separator"], windowGenerator,
                                 outputGeneratorTag))

    # Training Model
    model.train([trainSupervisedBatch, trainUnsupervisedDatasetBatch], numEpochs, devReader,
                callbacks=callbacks)


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = JsonArgParser(UNSUPERVISED_BACKPROPAGATION_PARAMETERS).parse(sys.argv[1])
    main(**parameters)
