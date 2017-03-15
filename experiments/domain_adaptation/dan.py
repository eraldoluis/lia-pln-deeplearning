#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script roda um modelo, chamado de DAN,
    baseado no trabalho Unsupervised Domain Adaptation by Backpropagation.
"""


import importlib
import logging
import logging.config
import math
import os
import random
import sys

import numpy as np
import theano
import theano.tensor as T
from pandas import json

from args.JsonArgParser import JsonArgParser
from data.BatchIterator import SyncBatchIterator
from data.CapitalizationFeatureGenerator import CapitalizationFeatureGenerator
from data.CharacterWindowGenerator import CharacterWindowGenerator
from data.ConstantLabel import ConstantLabel
from data.Embedding import Embedding
from data.LabelGenerator import LabelGenerator
from data.Lexicon import Lexicon
from data.SuffixFeatureGenerator import SuffixFeatureGenerator
from data.TokenDatasetReader import TokenLabelReader, TokenReader
from data.WordWindowGenerator import WordWindowGenerator
from model.Callback import Callback
from model.DANModel import DANModel
from model.Metric import LossMetric, AccuracyMetric
from model.ModelWriter import ModelWriter
from model.Objective import NegativeLogLikelihood
from model.Prediction import ArgmaxPrediction
from nnet.ActivationLayer import ActivationLayer, softmax, tanh, sigmoid
from nnet.ConcatenateLayer import ConcatenateLayer
from nnet.EmbeddingConvolutionalLayer import EmbeddingConvolutionalLayer
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.FlattenLayer import FlattenLayer
from nnet.GradientReversalLayer import GradientReversalLayer
from nnet.LinearLayer import LinearLayer
from nnet.WeightGenerator import ZeroWeightGenerator, GlorotUniform, SigmoidGlorot
from optim.Adagrad import Adagrad
from optim.SGD import SGD
from persistence.H5py import H5py
from util.jsontools import dict2obj
from util.util import getFilters

UNSUPERVISED_BACKPROPAGATION_PARAMETERS = {
    "token_label_separator": {"required": True,
                              "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "word_filters": {"required": True,
                     "desc": "a list which contains the filters. Each filter is describe by your module name + . + class name"},

    # Filters
    "suffix_filters": {"default": [],
                       "desc": "a list which contains the filters that will be used to process the suffix."
                               "These filters will be applied in tokens and after that the program will get suffix of each token."
                               "Each filter is describe by your module name + . + class name"},
    "cap_filters": {"default": [],
                    "desc": "a list which contains the filters that will be used to process the capitalization. "
                            "These filters will be applied in tokens and after that the program will get capitalization of each token."
                            "Each filter is describe by your module name + . + class name"},
    "char_filters": {"default": [], "desc": "list contains the filters that will be used in character embedding. "
                                            "These filters will be applied in tokens and after that the program will get the characters of each token ."
                                            "Each filter is describe by your module name + . + class name"},

    "label_file": {"desc": "", "required": True},
    "batch_size": {"required": True},
    "lambda_gradient": {"desc": "", "required": True},

    "alpha": {"desc": "", "required": False},
    "height": {"default": 1, "desc": "", "required": False},
    "train_source": {"desc": "Supervised Training File Path"},
    "train_target": {"desc": "Unsupervised Training File Path"},
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},
    "lr": {"desc": "learning rate value"},
    "save_model": {"desc": "Path + basename that will be used to save the weights and embeddings to be saved."},

    "test": {"desc": "Test File Path"},
    "dev": {"desc": "Development File Path"},
    "aux_devs": {
        "desc": "The parameter 'dev' represents the main dev and this parameter represents the auxiliary devs that"
                " will be use to evaluate the model."},
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
    "normalization": {"desc": "Choose the normalize method to be applied on  word embeddings. "
                              "The possible values are: minmax or mean"},
    "activation_hidden_extractor": {"default": "tanh", "desc": "This parameter chooses the type of activation function"
                                                               " that will be used in the hidden layer of the extractor. Options: sigmoid or tanh"},

    "with_charwnn": {"default": False, "desc": "Enable or disable the charwnn of the model"},
    "conv_size": {"default": 50, "desc": "The number of neurons in the convolutional layer"},
    "char_emb_size": {"default": 10, "desc": "The size of char embedding"},
    "char_window_size": {"default": 5, "desc": "The size of character windows."},
    "charwnn_with_act": {"default": True,
                         "desc": "Enable or disable the use of a activation function in the convolution. "
                                 "When this parameter is true, we use tanh as activation function"},

    "with_hidden": {"default": True,
                    "desc": "If this parameter is False, so the hidden before the softmax layer is removed from NN."},

    # Hand-crafted features
    "cap_emb_size": {"default": 5, "desc": ""},
    "suffix_emb_size": {"default": 5, "desc": ""},
    "suffix_size": {"default": 0, "desc": ""},
    "use_capitalization": {"default": False, "desc": ""},
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


class DevCallback(Callback):
    def __init__(self, model, devs, tokenLabelSep, inputGenerators, outputGenerators):
        self.__datasetIterators = []
        self.__model = model

        for devFile in devs:
            devDatasetReader = TokenLabelReader(devFile, tokenLabelSep)
            devIterator = SyncBatchIterator(devDatasetReader, inputGenerators, outputGenerators, sys.maxint,
                                            shuffle=False)

            self.__datasetIterators.append(devIterator)

    def onEpochEnd(self, epoch, logs={}):
        for it in self.__datasetIterators:
            self.__model.test(it)



def main(args):
    log = logging.getLogger(__name__)
    log.info(args)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    parametersToSaveOrLoad = {"word_filters", "suffix_filters", "char_filters", "cap_filters",
                              "alg", "hidden_activation_function", "word_window_size", "char_window_size",
                              "hidden_size_unsupervised_part", "hidden_size_supervised_part", "with_charwnn",
                              "conv_size", "charwnn_with_act", "suffix_size", "use_capitalization",
                              "start_symbol", "end_symbol", "with_hidden"}

    # Load parameters of the saving model
    if args.load_model:
        persistentManager = H5py(args.load_model)
        savedParameters = json.loads(persistentManager.getAttribute("parameters"))

        if savedParameters.get("charwnn_filters", None) != None:
            savedParameters["char_filters"] = savedParameters["charwnn_filters"]
            savedParameters.pop("charwnn_filters")
            print savedParameters

        log.info("Loading parameters of the model")
        args = args._replace(**savedParameters)

    log.info(str(args))

    wordWindowSize = args.word_window_size
    hiddenLayerSize = args.hidden_size
    batchSize = args.batch_size
    startSymbol = args.start_symbol
    endSymbol = args.end_symbol
    numEpochs = args.num_epochs
    lr = args.lr
    _lambda = theano.shared(args.lambda_gradient, "lambda")
    # _lambda = theano.shared(0.0, "lambda")
    useAdagrad = args.adagrad
    shuffle = args.shuffle
    supHiddenLayerSize = args.hidden_size_supervised_part
    unsupHiddenLayerSize = args.hidden_size_unsupervised_part
    normalizeMethod = args.normalization.lower() if args.normalization is not None else None
    activationHiddenExtractor = args.activation_hidden_extractor
    hiddenActFunctionName = args.hidden_activation_function

    withCharWNN = args.with_charwnn
    charEmbeddingSize = args.char_emb_size
    charWindowSize = args.char_window_size
    startSymbolChar = "</s>"

    suffixEmbSize = args.suffix_emb_size
    capEmbSize = args.cap_emb_size

    useSuffixFeatures = args.suffix_size > 0
    useCapFeatures = args.use_capitalization

    # Insert the character that will be used to fill the matrix
    # with a dimension lesser than chosen dimension.This enables that the convolution is performed by a matrix multiplication.
    artificialChar = "ART_CHAR"

    # TODO: the maximum number of characters of word is fixed in 20.
    numMaxChar = 20

    # Lendo Filtros do wnn
    log.info("Lendo filtros básicos")
    wordFilters = getFilters(args.word_filters, log)

    # Lendo Filtros do charwnn
    log.info("Lendo filtros do charwnn")
    charFilters = getFilters(args.char_filters, log)

    # Lendo Filtros do suffix
    log.info("Lendo filtros do sufixo")
    suffixFilters = getFilters(args.suffix_filters, log)

    # Lendo Filtros da capitalização
    log.info("Lendo filtros da capitalização")
    capFilters = getFilters(args.cap_filters, log)

    if withCharWNN and (useSuffixFeatures or useCapFeatures):
        raise Exception("It's impossible to use hand-crafted features with Charwnn.")

    # Read word lexicon and create word embeddings
    if args.load_model:
        wordLexicon = Lexicon.fromPersistentManager(persistentManager, "word_lexicon")
        vectors = EmbeddingLayer.getEmbeddingFromPersistenceManager(persistentManager, "word_embedding_layer")

        wordEmbedding = Embedding(wordLexicon, vectors)

    elif args.word_embedding:
        wordLexicon, wordEmbedding = Embedding.fromWord2Vec(args.word_embedding, "UUUNKKK", "word_lexicon")
    # elif args.word_lexicon:
    #     wordLexicon = Lexicon.fromTextFile(args.word_lexicon, True, "word_lexicon")
    #     wordEmbedding = Embedding(wordLexicon, vectors=None, embeddingSize=embeddingSize)
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
            charLexicon = Lexicon.fromTextFile(args.char_lexicon, True, "char_lexicon")
            charEmbedding = Embedding(charLexicon, vectors=None, embeddingSize=charEmbeddingSize)
        else:
            log.error("You need to set one of these parameters: load_model or char_lexicon")
            return
    else:
        # Read suffix lexicon if suffix size is greater than 0
        if useSuffixFeatures:
            if args.load_model:
                suffixLexicon = Lexicon.fromPersistentManager(persistentManager, "suffix_lexicon")
                vectors = EmbeddingConvolutionalLayer.getEmbeddingFromPersistenceManager(persistentManager,
                                                                                         "suffix_embedding")

                suffixEmbedding = Embedding(suffixLexicon, vectors)
            elif args.suffix_lexicon:
                suffixLexicon = Lexicon.fromTextFile(args.suffix_lexicon, True, "suffix_lexicon")
                suffixEmbedding = Embedding(suffixLexicon, vectors=None, embeddingSize=suffixEmbSize)
            else:
                log.error("You need to set one of these parameters: load_model or suffix_lexicon")
                return

        # Read capitalization lexicon
        if useCapFeatures:
            if args.load_model:
                capLexicon = Lexicon.fromPersistentManager(persistentManager, "cap_lexicon")
                vectors = EmbeddingConvolutionalLayer.getEmbeddingFromPersistenceManager(persistentManager,
                                                                                         "cap_embedding")

                capEmbedding = Embedding(capLexicon, vectors)
            elif args.cap_lexicon:
                capLexicon = Lexicon.fromTextFile(args.cap_lexicon, True, "cap_lexicon")
                capEmbedding = Embedding(capLexicon, vectors=None, embeddingSize=capEmbSize)
            else:
                log.error("You need to set one of these parameters: load_model or cap_lexicon")
                return

    # Read labels
    if args.load_model:
        labelLexicon = Lexicon.fromPersistentManager(persistentManager, "label_lexicon")
    elif args.label_file:
        labelLexicon = Lexicon.fromTextFile(args.label_file, False, lexiconName="label_lexicon")
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

    if withCharWNN and (useSuffixFeatures or useCapFeatures):
        raise Exception("It's impossible to use hand-crafted features with Charwnn.")

    if args.decay.lower() == "normal":
        decay = 0.0
    elif args.decay.lower() == "divide_epoch":
        decay = 1.0

    # Add the lexicon of target
    domainLexicon = Lexicon(None)

    domainLexicon.put("0")
    domainLexicon.put("1")
    domainLexicon.stopAdd()

    # Build neural network
    wordWindow = T.lmatrix("word_window")
    supervisedLabel = T.lvector("supervisedLabel")
    unsupervisedLabel = T.lvector("unsupervisedLabel")

    inputModel = [wordWindow]

    wordEmbeddingLayer = EmbeddingLayer(wordWindow, wordEmbedding.getEmbeddingMatrix(), trainable=True,
                                        name="word_embedding_layer")
    flatten = FlattenLayer(wordEmbeddingLayer)

    if withCharWNN:
        # Use the convolution
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
    elif useSuffixFeatures or useCapFeatures:
        # Use hand-crafted features
        concatenateInputs = [flatten]
        nmFetauresByWord = wordEmbedding.getEmbeddingSize()

        if useSuffixFeatures:
            log.info("Using suffix features")

            suffixInput = T.lmatrix("suffix_input")
            suffixEmbLayer = EmbeddingLayer(suffixInput, suffixEmbedding.getEmbeddingMatrix(),
                                            name="suffix_embedding")
            suffixFlatten = FlattenLayer(suffixEmbLayer)
            concatenateInputs.append(suffixFlatten)

            nmFetauresByWord += suffixEmbedding.getEmbeddingSize()
            inputModel.append(suffixInput)

        if useCapFeatures:
            log.info("Using capitalization features")

            capInput = T.lmatrix("capitalization_input")
            capEmbLayer = EmbeddingLayer(capInput, capEmbedding.getEmbeddingMatrix(),
                                         name="cap_embedding")
            capFlatten = FlattenLayer(capEmbLayer)
            concatenateInputs.append(capFlatten)

            nmFetauresByWord += capEmbedding.getEmbeddingSize()
            inputModel.append(capInput)

        layerBeforeLinear = ConcatenateLayer(concatenateInputs)
        sizeLayerBeforeLinear = wordWindowSize * nmFetauresByWord
    else:
        # Use only the word embeddings
        layerBeforeLinear = flatten
        sizeLayerBeforeLinear = wordWindowSize * wordEmbedding.getEmbeddingSize()

    # The rest of the NN
    if args.with_hidden:
        hiddenActFunction = method_name(hiddenActFunctionName)
        weightInit = SigmoidGlorot() if hiddenActFunction == sigmoid else GlorotUniform()

        linear1 = LinearLayer(layerBeforeLinear, sizeLayerBeforeLinear, hiddenLayerSize,
                              weightInitialization=weightInit, name="linear1")
        act1 = ActivationLayer(linear1, hiddenActFunction)

        layerBeforeSoftmax = act1
        sizeLayerBeforeSoftmax = hiddenLayerSize
        log.info("Using hidden layer")
    else:
        layerBeforeSoftmax = layerBeforeLinear
        sizeLayerBeforeSoftmax = sizeLayerBeforeLinear
        log.info("Not using hidden layer")

    supervisedBatches = layerBeforeSoftmax.getOutput()[:supervisedLabel.shape[0]]

    supervisedLinear = LinearLayer(supervisedBatches, sizeLayerBeforeSoftmax, labelLexicon.getLen(),
                                   weightInitialization=ZeroWeightGenerator(),
                                   name="linear_softmax")
    supervisedSoftmax = ActivationLayer(supervisedLinear, softmax)

    # Create the layers with the domain classifier
    gradientReversalSource = GradientReversalLayer(layerBeforeLinear, _lambda)

    unsupervisedLinear = LinearLayer(gradientReversalSource, sizeLayerBeforeLinear, domainLexicon.getLen(),
                                     weightInitialization=ZeroWeightGenerator(), name="linear_softmax_unsupervised")

    unsupervisedSoftmax = ActivationLayer(unsupervisedLinear, softmax)

    # Set loss and prediction and retrieve all layers
    supervisedOutput = supervisedSoftmax.getOutput()
    supervisedPrediction = ArgmaxPrediction(1).predict(supervisedOutput)
    supervisedLoss = NegativeLogLikelihood().calculateError(supervisedOutput, supervisedPrediction, supervisedLabel)

    unsupervisedOutput = unsupervisedSoftmax.getOutput()
    unsupervisedPred = ArgmaxPrediction(1).predict(unsupervisedOutput)
    unsupervisedLoss = NegativeLogLikelihood().calculateError(unsupervisedOutput, None, unsupervisedLabel)

    # Load the model
    if args.load_model:
        alreadyLoaded = set([wordEmbeddingLayer])

        for o in ((unsupervisedSoftmax.getLayerSet() | supervisedSoftmax.getLayerSet()) - alreadyLoaded):
            if o.getName():
                persistentManager.load(o)

    # Set the input and output
    inputGenerators = [WordWindowGenerator(wordWindowSize, wordLexicon, wordFilters, startSymbol, endSymbol)]

    if withCharWNN:
        inputGenerators.append(
            CharacterWindowGenerator(charLexicon, numMaxChar, charWindowSize, wordWindowSize, artificialChar,
                                     startSymbolChar, startPaddingWrd=startSymbol, endPaddingWrd=endSymbol,
                                     filters=charFilters))
    else:
        if useSuffixFeatures:
            inputGenerators.append(
                SuffixFeatureGenerator(args.suffix_size, wordWindowSize, suffixLexicon, suffixFilters))

        if useCapFeatures:
            inputGenerators.append(CapitalizationFeatureGenerator(wordWindowSize, capLexicon, capFilters))

    outputGeneratorLabel = LabelGenerator(labelLexicon)
    unsupervisedLabelSource = ConstantLabel(domainLexicon, "0")
    unsupervisedLabelTarget = ConstantLabel(domainLexicon, "1")

    log.info("Reading training examples")

    if args.train_source:
        # Reading supervised and unsupervised data sets.
        trainSupervisedDatasetReader = TokenLabelReader(args.train_source, args.token_label_separator)
        trainSupervisedBatch = SyncBatchIterator(trainSupervisedDatasetReader, inputGenerators,
                                                 [outputGeneratorLabel, unsupervisedLabelSource], batchSize[0],
                                                 shuffle=shuffle)

        # Get Unsupervised Input
        trainUnsupervisedDatasetReader = TokenReader(args.train_target)
        trainUnsupervisedDatasetBatch = SyncBatchIterator(trainUnsupervisedDatasetReader,
                                                          inputGenerators,
                                                          [unsupervisedLabelTarget], batchSize[1], shuffle=shuffle)

    # Printing embedding information
    dictionarySize = wordEmbedding.getNumberOfVectors()

    log.info("Size of  word dictionary and word embedding size: %d and %d" % (
        dictionarySize, wordEmbedding.getEmbeddingSize()))

    if withCharWNN:
        log.info("Size of  char dictionary and char embedding size: %d and %d" % (
            charEmbedding.getNumberOfVectors(), charEmbedding.getEmbeddingSize()))

    if useSuffixFeatures:
        log.info("Size of  suffix dictionary and suffix embedding size: %d and %d" % (
            suffixEmbedding.getNumberOfVectors(), suffixEmbedding.getEmbeddingSize()))

    if useCapFeatures:
        log.info("Size of  capitalization dictionary and capitalization embedding size: %d and %d" % (
            capEmbedding.getNumberOfVectors(), capEmbedding.getEmbeddingSize()))

    # Creates model
    if useAdagrad:
        log.info("Using ADAGRAD")
        opt = Adagrad(lr=lr, decay=decay)
    else:
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)

    allLayers = supervisedSoftmax.getLayerSet() | unsupervisedSoftmax.getLayerSet()

    supTrainMetrics = [
        LossMetric("TrainSupervisedLoss", supervisedLoss),
        AccuracyMetric("TrainSupervisedAcc", supervisedLabel, supervisedPrediction),
        LossMetric("TrainUnsupervisedLoss", unsupervisedLoss),
    ]

    unsTrainMetrics = [
        LossMetric("TrainUnsupervisedLoss", unsupervisedLoss),
        AccuracyMetric("TrainUnsupervisedAccuracy", unsupervisedLabel, unsupervisedPred)
    ]

    evalMetrics = [
        AccuracyMetric("EvalAcc", supervisedLabel, supervisedPrediction)
    ]

    testMetrics = [AccuracyMetric("TestAcc", supervisedLabel, supervisedPrediction)]

    model = DANModel(inputModel, supervisedLabel, unsupervisedLabel,
                     allLayers, opt, supervisedPrediction, supervisedLoss,
                     unsupervisedLoss, supTrainMetrics, unsTrainMetrics, evalMetrics, testMetrics,
                     mode=None)

    # Get dev inputs and output
    if args.dev:
        log.info("Reading development examples")
        devDatasetReader = TokenLabelReader(args.dev, args.token_label_separator)
        devReader = SyncBatchIterator(devDatasetReader, inputGenerators, [outputGeneratorLabel], sys.maxint,
                                      shuffle=False)

    if args.train_source:
        callbacks = []
        log.info("Usando lambda fixo: " + str(_lambda.get_value()))
        # log.info("Usando lambda variado. alpha=" + str(args.alpha) + " height=" + str(args.height))
        # callbacks.append(ChangeLambda(_lambda, args.alpha, numEpochs, args.height))

        if args.save_model:
            savePath = args.save_model
            objsToSave = list(supervisedSoftmax.getLayerSet() | unsupervisedSoftmax.getLayerSet()) + [wordLexicon,
                                                                                                      labelLexicon]

            if withCharWNN:
                objsToSave.append(charLexicon)

            if useSuffixFeatures:
                objsToSave.append(suffixLexicon)

            if useCapFeatures:
                objsToSave.append(capLexicon)

            modelWriter = ModelWriter(savePath, objsToSave, args, parametersToSaveOrLoad)

        if args.aux_devs:
            callbacks.append(
                DevCallback(model, args.aux_devs, args.token_label_separator, inputGenerators, [outputGeneratorLabel]))

        log.info("Training")
        # Training Model
        model.train([trainSupervisedBatch, trainUnsupervisedDatasetBatch], numEpochs, devReader,
                    callbacks=callbacks)

        if args.save_model:
            modelWriter.save()

            # Testing
    if args.test:
        if isinstance(args.test, (basestring, unicode)):
            tests = [args.test]
        else:
            tests = args.test

        for test in tests:
            log.info("Reading test examples")
            testDatasetReader = TokenLabelReader(test, args.token_label_separator)
            testReader = SyncBatchIterator(testDatasetReader, inputGenerators, [outputGeneratorLabel], sys.maxint,
                                           shuffle=False)

            log.info("Testing")
            model.test(testReader)


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

    parameters = dict2obj(JsonArgParser(UNSUPERVISED_BACKPROPAGATION_PARAMETERS).parse(sys.argv[1]))
    main(parameters)
