#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import logging
import logging.config
import numpy as np
import os
import random
import sys

import theano.tensor as T

from args.JsonArgParser import JsonArgParser
from data.BatchIterator import SyncBatchIterator
from data.CharacterWindowGenerator import CharacterWindowGenerator
from data.Embedding import Embedding
from data.LabelGenerator import LabelGenerator
from data.Lexicon import Lexicon
from data.TokenDatasetReader import TokenLabelPerLineReader
from data.WordWindowGenerator import WordWindowGenerator
from model.BasicModel import BasicModel
from model.Metric import LossMetric, AccuracyMetric, FMetric, CustomMetric
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
from util.jsontools import dict2obj

WNN_PARAMETERS = {
    "word_filters": {
        "required": False,
        "desc": "a list which contains the filters. Each filter is describe by your module name + . + class name"
    },
    "label_file": {
        "required": True,
        "desc": "file with all possible labels"
    },
    "word_lexicon": {
        "desc": "word lexicon"
    },
    "char_lexicon": {
        "desc": "char lexicon"
    },

    # Datasets.
    "train": {
        "required": True,
        "desc": "Training File Path"
    },
    "dev": {
        "desc": "Development File Path"
    },
    "test": {
        "desc": "Test File Path"
    },

    # Training parameters
    "lr": {"desc": "learning rate value"},
    "adagrad": {"desc": "Activate AdaGrad updates.", "default": True},
    "batch_size": {"default": 1},
    "decay": {"default": "DIVIDE_EPOCH",
              "desc": "Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available"},
    "hidden_activation_function": {"default": "tanh",
                                   "desc": "the activation function of the hidden layer. The possible values are: tanh and sigmoid"},
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},
    "shuffle": {"default": True, "desc": "enable the shuffle of training examples."},

    # Basic NN parameters
    "normalization": {"desc": "Choose the normalize method to be applied on word embeddings. "
                              "The possible values are: minmax or mean"},
    "normFactor": {"desc": "Factor to be multiplied by the normalized word vectors."},
    "hidden_size": {"default": 300, "desc": "The number of neurons in the hidden layer"},
    "word_emb_size": {"default": 100, "desc": "size of word embedding"},
    "word_embedding": {"desc": "word embedding File Path"},
    "word_window_size": {"default": 5, "desc": "The size of words for the wordsWindow"},

    "conv_size": {"default": 50, "desc": "The number of neurons in the convolutional layer"},
    "char_emb_size": {"default": 10, "desc": "The size of char embedding"},
    "char_window_size": {"default": 5, "desc": "The size of character windows."},

    # Other parameter
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc": "Seed used in the random number generator"},
}


def mainWnnNer(args):
    # Initializing parameters.
    log = logging.getLogger(__name__)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    log.info(str(args))
    
    # GPU configuration.
    log.info("floatX=", theano.config.floatX)
    log.info("device=", theano.config.device)

    # Parameters.
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
    batchSize = args.batch_size

    charEmbeddingSize = args.char_emb_size
    charWindowSize = args.char_window_size
    charConvSize = args.conv_size

    # Word filters.
    log.info("Loading word filters...")
    wordFilters = getFilters(args.word_filters, log)

    # Loading word lexicon or word embedding.
    if args.word_embedding:
        log.info("Loading word embedding...")
        wordLexicon, wordEmbedding = Embedding.fromWord2Vec(args.word_embedding, "UUUNKKK", "word_lexicon")
    elif args.word_lexicon:
        log.info("Loading word lexicon...")
        wordLexicon = Lexicon.fromTextFile(args.word_lexicon, True, "word_lexicon")
        wordEmbedding = Embedding(wordLexicon, vectors=None, embeddingSize=embeddingSize)
    else:
        log.error("You need to set one of these parameters: load_model, word_embedding or word_lexicon")
        sys.exit(1)

    # Loading char lexicon
    if args.char_lexicon:
        log.info("Loading char lexicon...")
        charLexicon = Lexicon.fromTextFile(args.char_lexicon, True, "char_lexicon")
        charEmbedding = Embedding(charLexicon, vectors=None, embeddingSize=charEmbeddingSize)
    else:
        log.error("You need to set the parameter: char_lexicon")
        sys.exit(1)

    # Loading label lexicon.
    if args.label_file:
        log.info("Loading label lexicon...")
        labelLexicon = Lexicon.fromTextFile(args.label_file, False, lexiconName="label_lexicon")
    else:
        log.error("You need to set one of these parameters: load_model, word_embedding or word_lexicon")
        sys.exit(1)

    # Normalize the word embedding
    if normalizeMethod is not None:
        normFactor = 1
        if args.normFactor is not None:
            normFactor = args.normFactor

        if normalizeMethod == "minmax":
            log.info("Normalizing word embedding: minmax")
            wordEmbedding.minMaxNormalization(norm_coef=normFactor)
        elif normalizeMethod == "mean":
            log.info("Normalizing word embedding: mean")
            wordEmbedding.meanNormalization(norm_coef=normFactor)
        else:
            log.error("Unknown normalization method: %s" % normalizeMethod)
            sys.exit(1)
    elif args.normFactor is not None:
        log.error("Parameter normFactor cannot be present without normalization.")
        sys.exit(1)

    dictionarySize = wordEmbedding.getNumberOfVectors()
    log.info("Size of word lexicon is %d and word embedding size is %d" % (dictionarySize, embeddingSize))

    # Setup the input and (golden) output generators (readers).
    inputGenerators = [
        WordWindowGenerator(wordWindowSize, wordLexicon, wordFilters, startSymbol, endSymbol),
        CharacterWindowGenerator(charLexicon, 20, charWindowSize, wordWindowSize, "ART_CHAR", "</s>",
                                 startPaddingWrd=startSymbol, endPaddingWrd=endSymbol,
                                 filters=getFilters([], log))
    ]
    outputGenerator = LabelGenerator(labelLexicon)

    log.info("Reading training examples")

    trainDatasetReader = TokenLabelPerLineReader(args.train, labelTknSep='\t')
    trainReader = SyncBatchIterator(trainDatasetReader, inputGenerators, [outputGenerator], batchSize, shuffle=shuffle)

    # Build neural network
    wordWindow = T.lmatrix("word_window")
    inputModel = [wordWindow]

    wordEmbeddingLayer = EmbeddingLayer(wordWindow, wordEmbedding.getEmbeddingMatrix(), trainable=True,
                                        name="word_embedding_layer")
    flatWordEmbedding = FlattenLayer(wordEmbeddingLayer)

    charWindowIdxs = T.ltensor4(name="char_window_idx")
    inputModel.append(charWindowIdxs)

    # # TODO: debug
    # theano.config.compute_test_value = 'warn'
    # ex = trainIterator.next()
    # inWords.tag.test_value = ex[0][0]
    # outLabel.tag.test_value = ex[1][0]

    charEmbeddingConvLayer = EmbeddingConvolutionalLayer(charWindowIdxs, charEmbedding.getEmbeddingMatrix(), 20,
                                                         charConvSize, charWindowSize, charEmbeddingSize, tanh,
                                                         name="char_convolution_layer")

    layerBeforeLinear = ConcatenateLayer([flatWordEmbedding, charEmbeddingConvLayer])
    sizeLayerBeforeLinear = wordWindowSize * (wordEmbedding.getEmbeddingSize() + charConvSize)

    hiddenActFunction = method_name(hiddenActFunctionName)
    weightInit = SigmoidGlorot() if hiddenActFunction == sigmoid else GlorotUniform()

    linearHidden = LinearLayer(layerBeforeLinear, sizeLayerBeforeLinear, hiddenLayerSize,
                               weightInitialization=weightInit,
                               name="linear1")
    actHidden = ActivationLayer(linearHidden, hiddenActFunction)

    linearSoftmax = LinearLayer(actHidden, hiddenLayerSize, labelLexicon.getLen(),
                                weightInitialization=ZeroWeightGenerator(),
                                name="linear_softmax")
    actSoftmax = ActivationLayer(linearSoftmax, softmax)
    prediction = ArgmaxPrediction(1).predict(actSoftmax.getOutput())

    # Get dev inputs and (golden) outputs.
    if args.dev is not None:
        log.info("Reading development examples")
        devDatasetReader = TokenLabelPerLineReader(args.dev, labelTknSep='\t')
        devReader = SyncBatchIterator(devDatasetReader, inputGenerators, [outputGenerator], sys.maxint, shuffle=False)
    else:
        devReader = None

    # Output symbolic tensor variable.
    y = T.lvector("y")

    if args.decay.lower() == "normal":
        decay = 0.0
    elif args.decay.lower() == "divide_epoch":
        decay = 1.0

    if args.adagrad:
        log.info("Training algorithm: Adagrad")
        opt = Adagrad(lr=lr, decay=decay)
    else:
        log.info("Training algorithm: SGD")
        opt = SGD(lr=lr, decay=decay)

    # Training loss function.
    loss = NegativeLogLikelihood().calculateError(actSoftmax.getOutput(), prediction, y)

    # # TODO: debug
    # opt.lr.tag.test_value = 0.02

    # Metrics.
    trainMetrics = [
        LossMetric("LossTrain", loss, True),
        AccuracyMetric("AccTrain", y, prediction),
    ]

    evalMetrics = [
        LossMetric("LossDev", loss, True),
        AccuracyMetric("AccDev", y, prediction),
        FMetric("FMetricDev", y, prediction),
        CustomMetric("CustomMetricDev", y, prediction),
    ]

    testMetrics = [
        LossMetric("LossTest", loss, True),
        AccuracyMetric("AccTest", y, prediction),
        FMetric("FMetricTest", y, prediction),
        CustomMetric("CustomMetricTest", y, prediction),
    ]

    log.info("Compiling the network...")
    # # TODO: debug
    # mode = theano.compile.debugmode.DebugMode(optimizer=None)
    mode = None
    wnnModel = BasicModel(inputModel, [y], actSoftmax.getLayerSet(), opt, prediction, loss, trainMetrics=trainMetrics,
                          evalMetrics=evalMetrics, testMetrics=testMetrics, mode=mode)

    log.info("Training...")
    wnnModel.train(trainReader, numEpochs, devReader)

    # Testing.
    if args.test:
        log.info("Reading test dataset...")
        testDatasetReader = TokenLabelPerLineReader(args.test, labelTknSep='\t')
        testReader = SyncBatchIterator(testDatasetReader, inputGenerators, [outputGenerator], sys.maxint, shuffle=False)

        log.info("Testing...")
        wnnModel.test(testReader)


def getFilters(param, log):
    filters = []

    for filterName in param:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    return filters


def method_name(hiddenActFunction):
    if hiddenActFunction == "tanh":
        return tanh
    elif hiddenActFunction == "sigmoid":
        return sigmoid
    else:
        raise Exception("'hidden_activation_function' value don't valid.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Syntax error!\n')
        sys.stderr.write('\tExpected argument: <JSON config file>\n')
        sys.exit(1)

    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = dict2obj(JsonArgParser(WNN_PARAMETERS).parse(sys.argv[1]))
    mainWnnNer(parameters)
