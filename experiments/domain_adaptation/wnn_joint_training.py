#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script trains .
"""
# TODO: comentar

import logging
import logging.config
import os
import random
import sys
from itertools import izip

import numpy as np
import theano.tensor as T
from pandas import json

from args.JsonArgParser import JsonArgParser
from data import BatchIteratorUnion
from data.BatchIterator import SyncBatchIterator
from data.CapitalizationFeatureGenerator import CapitalizationFeatureGenerator
from data.CharacterWindowGenerator import CharacterWindowGenerator
from data.Embedding import Embedding
from data.LabelGenerator import LabelGenerator
from data.Lexicon import Lexicon
from data.SuffixFeatureGenerator import SuffixFeatureGenerator
from data.TokenDatasetReader import TokenLabelReader
from data.WordWindowGenerator import WordWindowGenerator
from model.Callback import DevCallback
from model.JointMultiTaskModel import JointMultiTaskModel
from model.Metric import LossMetric, AccuracyMetric
from model.ModelWriter import ModelWriter
from model.Objective import NegativeLogLikelihood
from model.Prediction import ArgmaxPrediction
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
from util.util import getFilters

JOINT_TRAINING_PARAMETERS = {
    # Required parameters
    "token_label_separator": {"required": True,
                              "desc": "specify the character used to separate the token from the label in the dataset."},
    "predict_idx": {"desc": "Softmax index that will be used to predict a new example.",
                          "required": True},

    # General Parameters
    "seed": {"desc": "seed"},
    "word_filters": {"required": False,
                     "desc": "a list which contains the filters. Each filter is describe by your module name + . + class name"},
    "label_file": {"desc": "a list of files with labels of each task.",
                   "required": False},
    "training_file": {"desc": "List of file path names. We will use these files for train the NN."},
    "test": {"desc": "File path name or list of file path name. Those files will be used for evaluate our model."},
    "dev": {"desc": "File path name. Those files will be used for evaluate our model in each epoch."},
    "num_tasks": {"required": False,
                     "desc": "Total number of tasks. You need to set this parameter when you are training a model."},


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

    # Training parameters
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},
    "lr": {"desc": "learning rate value"},
    "save_model": {
        "desc": "File path name which we will save the model parameters and others informations about the model."},
    "shuffle": {"default": True, "desc": "able or disable the shuffle of training examples."},
    "aux_devs": {
        "desc": "The parameter 'dev' represents the main dev and this parameter represents the auxiliary devs that"
                " will be use to evaluate the model."},

    "adagrad": {"desc": "Activate AdaGrad updates.", "default": True},
    "decay": {"default": "DIVIDE_EPOCH",
              "desc": "Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available"},
    # "batch_size": {"required": False,
    #                "desc": "Batch size"},
    "load_model": {"desc": "File path name which stores NN parameters and others informations about the model."},

    # WNN parameters
    "alg": {"default": "window_word",
            "desc": "The type of algorithm to train and test. The posible inputs are: window_word or window_stn"},
    "with_hidden": {"default": True,
                    "desc": "If this parameter is False, so the hidden before the softmax layer is removed from NN."},
    "hidden_size": {"default": 300, "desc": "The number of neurons in the hidden layer"},
    "word_window_size": {"default": 5, "desc": "The size of words for the wordsWindow"},
    "word_emb_size": {"default": 100, "desc": "size of word embedding"},
    "word_embedding": {"desc": "word embedding file path name"},
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
    "hidden_activation_function": {"default": "tanh",
                                   "desc": "the activation function of the hidden layer. The possible values are: tanh and sigmoid"},
    "normalization": {"desc": "Choose the normalize method to be applied on  word embeddings. "
                              "The possible values are: minmax or mean"},

    # Charwnn parameters
    "with_charwnn": {"default": False, "desc": "Enable or disable the charwnn of the model"},
    "conv_size": {"default": 50, "desc": "The number of neurons in the convolutional layer"},
    "char_emb_size": {"default": 10, "desc": "The size of char embedding"},
    "char_window_size": {"default": 5, "desc": "The size of character windows."},
    "charwnn_with_act": {"default": True,
                         "desc": "Enable or disable the use of a activation function in the convolution. "
                                 "When this parameter is true, we use tanh as activation function"},

    # Hand-crafted features
    "cap_emb_size": {"default": 5, "desc": ""},
    "suffix_emb_size": {"default": 5, "desc": ""},
    "suffix_size": {"default": 0, "desc": ""},
    "use_capitalization": {"default": False, "desc": ""},

}


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
                              "start_symbol", "end_symbol", "with_hidden", "predict_idx", 'num_tasks'}

    # Load model parameters
    if args.load_model:
        persistentManager = H5py(args.load_model)
        savedParameters = json.loads(persistentManager.getAttribute("parameters"))

        log.info("Loading parameters of the model")
        args = args._replace(**savedParameters)

    # Print parameters
    log.info(str(args))

    # Initialize some variables
    wordWindowSize = args.word_window_size
    hiddenLayerSize = args.hidden_size
    # batchSize = args.batch_size
    batchSize = 1
    startSymbol = args.start_symbol
    endSymbol = args.end_symbol
    numEpochs = args.num_epochs
    lr = args.lr
    useAdagrad = args.adagrad
    shuffle = args.shuffle
    normalizeMethod = args.normalization.lower() if args.normalization is not None else None
    hiddenActFunctionName = args.hidden_activation_function

    withCharWNN = args.with_charwnn
    charEmbeddingSize = args.char_emb_size
    charWindowSize = args.char_window_size
    startSymbolChar = "</s>"

    suffixEmbSize = args.suffix_emb_size
    capEmbSize = args.cap_emb_size

    useSuffixFeatures = args.suffix_size > 0
    useCapFeatures = args.use_capitalization
    idxSoftmaxPredict = args.predict_idx
    numTasks = args.num_tasks

    # Insert the character that will be used to fill the matrix with a dimension lesser than a chosen
    # dimension.This enables to perform convolution by a matrix multiplication.
    artificialChar = "ART_CHAR"

    # TODO: the maximum number of characters of word is fixed in 20.
    numMaxChar = 20

    # Load WNN Filters
    log.info("Lendo filtros básicos")
    wordFilters = getFilters(args.word_filters, log)

    # Load CharWNN Filters
    log.info("Lendo filtros do charwnn")
    charFilters = getFilters(args.char_filters, log)

    # Load Suffix Filters
    log.info("Lendo filtros do sufixo")
    suffixFilters = getFilters(args.suffix_filters, log)

    # Load Capitalization Filters
    log.info("Lendo filtros da capitalização")
    capFilters = getFilters(args.cap_filters, log)

    if withCharWNN and (useSuffixFeatures or useCapFeatures):
        raise Exception("It's impossible to use hand-crafted features with Charwnn.")

    # Load word lexicon and load/create word embeddings
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

    if withCharWNN:
        # Load char lexicon and create/load char embeddings
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
        # if suffix size is greater than 0 so suffix lexicon are loaded.
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

        # Load capitalization features
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

    # Load label lexicons of each task.
    labelLexiconList = []

    if args.load_model:
        for i in range(numTasks):
            labelLexiconList.append(Lexicon.fromPersistentManager(persistentManager, "label_lexicon_%d" % i))
    elif args.label_file:
        for i in range(numTasks):
            labelLexiconList.append(Lexicon.fromTextFile(args.label_file, False, lexiconName="label_lexicon_%d" % i))
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

    # Set decay
    if args.decay.lower() == "normal":
        decay = 0.0
    elif args.decay.lower() == "divide_epoch":
        decay = 1.0

    # Build neural network
    wordWindow = T.lmatrix("word_window")
    y = T.lvector("supervisedLabel")

    inputModel = [wordWindow]

    wordEmbeddingLayer = EmbeddingLayer(wordWindow, wordEmbedding.getEmbeddingMatrix(), trainable=True,
                                        name="word_embedding_layer")
    flatten = FlattenLayer(wordEmbeddingLayer)

    if withCharWNN:
        # Use convolution
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
        # Use only word embeddings
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

    softmaxLayerList = []

    for idx, labelLexicon in enumerate(labelLexiconList):
        supervisedLinear = LinearLayer(layerBeforeSoftmax, sizeLayerBeforeSoftmax, labelLexicon.getLen(),
                                       weightInitialization=ZeroWeightGenerator(),
                                       name="linear_softmax_%d" % (idx))
        softmaxLayerList.append(ActivationLayer(supervisedLinear, softmax))

    # Get all layers of each task
    taskLayerList = []
    allLayers = set()

    for softmaxLayer in softmaxLayerList:
        taskLayerList.append(softmaxLayer.getLayerSet())
        allLayers |= softmaxLayer.getLayerSet()


    # Set loss function
    lossList = []
    predictionList = []

    for softmaxLayer in softmaxLayerList:
        # Obs: Loglikelihood do not use prediction
        lossList.append(NegativeLogLikelihood().calculateError(softmaxLayer, None, y))
        predictionList.append(ArgmaxPrediction(1).predict(softmaxLayerList[idxSoftmaxPredict].getOuput()))

    # Global loss function
    # loss = T.prod(T.stack(lossList), taskVector.T)

    # Set prediction function
    mainPredictionFunc = predictionList[idxSoftmaxPredict]

    # Load the model
    if args.load_model:
        alreadyLoaded = set([wordEmbeddingLayer])

        for o in (allLayers - alreadyLoaded):
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

    outputGeneratorList = []

    for labelLexicon in labelLexiconList:
        outputGeneratorList.append(LabelGenerator(labelLexicon))

    log.info("Reading training examples")

    if args.training_file:
        # Reading supervised and unsupervised data sets.
        batchIteratorList = []

        for file, outputGenerator in izip(args.training_file, outputGeneratorList):
            log.info("Reading %s" % file)
            trainingDatasetReader = TokenLabelReader(file, args.token_label_separator)
            batchIteratorList.append(
                SyncBatchIterator(trainingDatasetReader, inputGenerators, [outputGenerator], batchSize,
                                  shuffle=shuffle))

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

    # Choose gradient descent optimization algorithms
    if useAdagrad:
        log.info("Using ADAGRAD")
        opt = Adagrad(lr=lr, decay=decay)
    else:
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)

    # trainingMetrics = [LossMetric("Loss", loss)]
    taskMetrics = []
    for taskLoss, taskPrediction in izip(lossList, predictionList):
        metrics = [LossMetric("Loss", taskLoss), AccuracyMetric("TrainSupervisedAcc", y, taskPrediction)]
        taskMetrics.append(metrics)

    evalMetrics = [
        AccuracyMetric("EvalAcc", y, mainPredictionFunc)
    ]

    testMetrics = [AccuracyMetric("TestAcc", y, mainPredictionFunc)]

    model = JointMultiTaskModel(inputModel, [y], taskLayerList, opt, mainPredictionFunc, lossList, taskMetrics,
                                evalMetrics, testMetrics, mode=None)

    # Get dev inputs and output
    if args.dev:
        log.info("Reading development examples")
        devDatasetReader = TokenLabelReader(args.dev, args.token_label_separator)
        devReader = SyncBatchIterator(devDatasetReader, inputGenerators, [y], sys.maxint,
                                      shuffle=False)

    if args.training_file:
        callbacks = []

        if args.save_model:
            savePath = args.save_model
            objsToSave = allLayers + [wordLexicon, labelLexiconList]

            if withCharWNN:
                objsToSave.append(charLexicon)

            if useSuffixFeatures:
                objsToSave.append(suffixLexicon)

            if useCapFeatures:
                objsToSave.append(capLexicon)

            modelWriter = ModelWriter(savePath, objsToSave, args, parametersToSaveOrLoad)

        if args.aux_devs:
            callbacks.append(
                DevCallback(model, args.aux_devs, args.token_label_separator, inputGenerators, [y]))


        # Join all dataset in one dataset
        iterator = BatchIteratorUnion(batchIteratorList)

        # Training Model
        log.info("Training")
        model.train(iterator, numEpochs, devReader, callbacks=callbacks)

        # Save model
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
                testReader = SyncBatchIterator(testDatasetReader, inputGenerators, [y], sys.maxint, shuffle=False)

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
    # Logging
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    # Read parameters
    parameters = dict2obj(JsonArgParser(JOINT_TRAINING_PARAMETERS).parse(sys.argv[1]))

    main(parameters)
