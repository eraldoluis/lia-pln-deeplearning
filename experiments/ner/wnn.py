#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import logging
import logging.config
import numpy as np
import os
import random
import sys

import theano
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
    "word_filters": {
        "desc": "a list which contains the filters. Each filter is describe by your module name + . + class name"
    },

    # Training parameters
    "lr": {"desc": "learning rate value"},
    "adagrad": {"desc": "Activate AdaGrad updates.", "default": True},
    "batch_size": {"default": 1},
    "decay": {"default": "DIVIDE_EPOCH", "desc": "Set the learning rate update strategy. " +
                                                 "NORMAL and DIVIDE_EPOCH are the options available"},
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},
    "shuffle": {"default": True, "desc": "enable the shuffle of training examples."},
    "struct_grad": {"default": True, "desc": "Structured gradient for the word embedding layer."},
    "char_struct_grad": {"default": True, "desc": "Structured gradient for the character embedding layer."},
    "l2": {"desc": "L2 regularization parameter (multiplier)."},
    "label_file": {"required": True, "desc": "file with all possible labels"},

    # Word embedding.
    "word_lexicon": {"desc": "word lexicon"},
    "word_emb_size": {"default": 100, "desc": "size of word embedding"},
    "word_embedding": {"desc": "word embedding File Path"},
    "normalization": {"desc": "Choose the normalize method to be applied on word embeddings. "
                              "The possible values are: minmax or mean"},
    "norm_factor": {"desc": "Factor to be multiplied by the normalized word vectors."},
    "word_window_size": {"default": 5, "desc": "The size of words for the wordsWindow"},

    # Character embedding.
    "char_lexicon": {"desc": "char lexicon", "required": True},
    "conv_size": {"default": 50, "desc": "The number of neurons in the convolutional layer"},
    "char_emb_size": {"default": 10, "desc": "The size of char embedding"},
    "char_window_size": {"default": 5, "desc": "The size of character windows."},

    # Hidden layer.
    "hidden_size": {"default": 300, "desc": "The number of neurons in the hidden layer"},
    "hidden_activation_function": {"default": "tanh",
                                   "desc": "the activation function of the hidden layer. " +
                                           "The possible values are: tanh, sigmoid, relu."},

    # Other parameters.
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc": "Seed used in the random number generator"},
    "cv": {"desc": "Cross-validation configuration."}
}


def mainWnnNer(args):
    # Initializing parameters.
    log = logging.getLogger(__name__)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    log.info({"type": "args", "args": args})

    # GPU configuration.
    log.info({"floatX": str(theano.config.floatX), "device": str(theano.config.device)})

    # Parameters.
    # lr = args.lr
    # startSymbol = args.start_symbol
    # endSymbol = args.end_symbol
    # numEpochs = args.num_epochs
    # shuffle = args.shuffle
    # normalization = args.normalization
    # wordWindowSize = args.word_window_size
    # hiddenLayerSize = args.hidden_size
    # hiddenActFunctionName = args.hidden_activation_function
    # embeddingSize = args.word_emb_size
    # batchSize = args.batch_size
    # structGrad = args.struct_grad
    # charStructGrad = args.char_struct_grad
    #
    # charEmbeddingSize = args.char_emb_size
    # charWindowSize = args.char_window_size
    # charConvSize = args.conv_size

    # Word filters.
    log.info("Loading word filters...")
    wordFilters = getFilters(args.word_filters, log)

    # Loading/creating word lexicon and word embedding.
    if args.word_embedding is not None:
        log.info("Loading word embedding...")
        wordLexicon, wordEmbedding = Embedding.fromWord2Vec(args.word_embedding, "UUUNKKK", "word_lexicon")
    elif args.word_lexicon is not None:
        log.info("Loading word lexicon...")
        wordLexicon = Lexicon.fromTextFile(args.word_lexicon, True, "word_lexicon")
        wordEmbedding = Embedding(wordLexicon, vectors=None, embeddingSize=args.word_emb_size)
    else:
        log.error("You need to set one of these parameters: load_model, word_embedding or word_lexicon")
        sys.exit(1)

    # Loading char lexicon.
    log.info("Loading char lexicon...")
    charLexicon = Lexicon.fromTextFile(args.char_lexicon, True, "char_lexicon")

    # Character embedding.
    charEmbedding = Embedding(charLexicon, vectors=None, embeddingSize=args.char_emb_size)

    # Loading label lexicon.
    log.info("Loading label lexicon...")
    labelLexicon = Lexicon.fromTextFile(args.label_file, False, lexiconName="label_lexicon")

    # Normalize the word embedding
    if args.normalization is not None:
        normFactor = 1
        if args.norm_factor is not None:
            normFactor = args.norm_factor

        if args.normalization == "minmax":
            log.info("Normalizing word embedding: minmax")
            wordEmbedding.minMaxNormalization(norm_coef=normFactor)
        elif args.normalization == "mean":
            log.info("Normalizing word embedding: mean")
            wordEmbedding.meanNormalization(norm_coef=normFactor)
        else:
            log.error("Unknown normalization method: %s" % args.normalization)
            sys.exit(1)
    elif args.normFactor is not None:
        log.error("Parameter norm_factor cannot be present without normalization.")
        sys.exit(1)

    dictionarySize = wordEmbedding.getNumberOfVectors()
    log.info("Size of word lexicon is %d and word embedding size is %d" % (dictionarySize, args.word_emb_size))

    # Setup the input and (golden) output generators (readers).
    inputGenerators = [
        WordWindowGenerator(args.word_window_size, wordLexicon, wordFilters, args.start_symbol, args.end_symbol),
        CharacterWindowGenerator(lexicon=charLexicon, numMaxChar=20, charWindowSize=args.char_window_size,
                                 wrdWindowSize=args.word_window_size, artificialChar="ART_CHAR", startPadding="</s>",
                                 startPaddingWrd=args.start_symbol, endPaddingWrd=args.end_symbol,
                                 filters=getFilters([], log))
    ]
    outputGenerator = LabelGenerator(labelLexicon)

    if args.cv is not None:
        log.info("Reading training examples...")
        trainIterator = SyncBatchIterator(TokenLabelPerLineReader(args.train, labelTknSep='\t'), inputGenerators,
                                          [outputGenerator], args.batch_size, shuffle=args.shuffle,
                                          numCVFolds=args.cv.numFolds)
        cvGenerators = trainIterator.getCVGenerators()
        iFold = 0
        numFolds = len(cvGenerators)
        for train, dev in cvGenerators:
            log.info({"cv": {"fold": iFold, "numFolds": numFolds}})
            trainNetwork(args, log, trainIterator=train, devIterator=dev, wordEmbedding=wordEmbedding,
                         charEmbedding=charEmbedding, borrow=False, labelLexicon=labelLexicon)
    else:
        log.info("Reading training examples...")
        trainIterator = SyncBatchIterator(TokenLabelPerLineReader(args.train, labelTknSep='\t'), inputGenerators,
                                          [outputGenerator], args.batch_size, shuffle=args.shuffle)

        # Get dev inputs and (golden) outputs.
        devIterator = None
        if args.dev is not None:
            log.info("Reading development examples")
            devIterator = SyncBatchIterator(TokenLabelPerLineReader(args.dev, labelTknSep='\t'), inputGenerators,
                                            [outputGenerator], sys.maxint, shuffle=False)

        trainNetwork(args, log, trainIterator, devIterator, wordEmbedding, charEmbedding, borrow=True,
                     labelLexicon=labelLexicon)

    # Testing.
    if args.test:
        log.info("Reading test dataset...")
        testIterator = SyncBatchIterator(TokenLabelPerLineReader(args.test, labelTknSep='\t'), inputGenerators,
                                         [outputGenerator], sys.maxint, shuffle=False)

        log.info("Testing...")
        wnnModel.test(testIterator)

    log.info("Done!")


def trainNetwork(args, log, trainIterator, devIterator, wordEmbedding, charEmbedding, borrow, labelLexicon):
    # Build neural network.
    wordWindow = T.lmatrix("word_window")
    inputModel = [wordWindow]

    wordEmbeddingLayer = EmbeddingLayer(wordWindow, wordEmbedding.getEmbeddingMatrix(), borrow=borrow,
                                        structGrad=args.struct_grad, trainable=True, name="word_embedding_layer")
    flatWordEmbedding = FlattenLayer(wordEmbeddingLayer)

    charWindowIdxs = T.ltensor4(name="char_window_idx")
    inputModel.append(charWindowIdxs)

    # # TODO: debug
    # theano.config.compute_test_value = 'warn'
    # ex = trainIterator.next()
    # inWords.tag.test_value = ex[0][0]
    # outLabel.tag.test_value = ex[1][0]

    charEmbeddingConvLayer = EmbeddingConvolutionalLayer(charWindowIdxs, charEmbedding.getEmbeddingMatrix(), 20,
                                                         args.conv_size, args.char_window_size, args.char_emb_size,
                                                         tanh, structGrad=args.char_struct_grad,
                                                         name="char_convolution_layer", borrow=borrow)

    layerBeforeLinear = ConcatenateLayer([flatWordEmbedding, charEmbeddingConvLayer])
    sizeLayerBeforeLinear = args.word_window_size * (wordEmbedding.getEmbeddingSize() + args.conv_size)

    hiddenActFunction = method_name(args.hidden_activation_function)
    weightInit = SigmoidGlorot() if hiddenActFunction == sigmoid else GlorotUniform()

    linearHidden = LinearLayer(layerBeforeLinear, sizeLayerBeforeLinear, args.hidden_size,
                               weightInitialization=weightInit, name="linear1")
    actHidden = ActivationLayer(linearHidden, hiddenActFunction)

    linearSoftmax = LinearLayer(actHidden, args.hidden_size, labelLexicon.getLen(),
                                weightInitialization=ZeroWeightGenerator(), name="linear_softmax")
    actSoftmax = ActivationLayer(linearSoftmax, softmax)
    prediction = ArgmaxPrediction(1).predict(actSoftmax.getOutput())

    # Output symbolic tensor variable.
    y = T.lvector("y")

    if args.decay.lower() == "normal":
        decay = 0.0
    elif args.decay.lower() == "divide_epoch":
        decay = 1.0
    else:
        log.error("Unknown decay argument: %s" % args.decay)
        sys.exit(1)

    if args.adagrad:
        log.info("Training algorithm: Adagrad")
        opt = Adagrad(lr=args.lr, decay=decay)
    else:
        log.info("Training algorithm: SGD")
        opt = SGD(lr=args.lr, decay=decay)

    # Training loss function.
    loss = NegativeLogLikelihood().calculateError(actSoftmax.getOutput(), prediction, y)

    # L2 regularization.
    if args.l2:
        loss += args.l2 * (T.sum(T.square(linearHidden.getParameters()[0])))

    # # TODO: debug
    # opt.lr.tag.test_value = 0.02

    # Metrics.
    trainMetrics = [
        LossMetric("LossTrain", loss, True),
        AccuracyMetric("AccTrain", y, prediction)
    ]

    evalMetrics = None
    if args.dev:
        evalMetrics = [
            LossMetric("LossDev", loss, True),
            AccuracyMetric("AccDev", y, prediction),
            CustomMetric("CustomMetricDev", y, prediction)
        ]

    testMetrics = None
    if args.test:
        testMetrics = [
            CustomMetric("CustomMetricTest", y, prediction)
        ]

    log.info("Compiling the network...")
    # # TODO: debug
    # mode = theano.compile.debugmode.DebugMode(optimizer=None)
    mode = None
    wnnModel = BasicModel(inputModel, [y], actSoftmax.getLayerSet(), opt, prediction, loss, trainMetrics=trainMetrics,
                          evalMetrics=evalMetrics, testMetrics=testMetrics, mode=mode)

    log.info("Training...")
    wnnModel.train(trainIterator, args.num_epochs, devIterator)


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
    if hiddenActFunction == "sigmoid":
        return sigmoid
    if hiddenActFunction == "relu":
        return T.nnet.relu
    # if hiddenActFunction == "hardtanh":
    #     return T.nnet.
    raise Exception("'hidden_activation_function' value don't valid.")


def main():
    nArgs = len(sys.argv)
    if nArgs != 2 and nArgs != 4:
        sys.stderr.write('Syntax error! Expected arguments: <params_file> [<num_folds> <params_dist>]\n')
        sys.stderr.write('\t<params_file>: JSON-formatted file containing parameter values\n')
        sys.stderr.write('\t<num_folds>: number of cross-validation folds used in random search\n')
        sys.stderr.write('\t<params_dist>: JSON-formatted file containing the parameter distributions used in random '
                         'search\n')
        sys.exit(1)

    full_path = os.path.realpath(__file__)
    path = os.path.split(full_path)[0]

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = dict2obj(JsonArgParser(WNN_PARAMETERS).parse(sys.argv[1]))
    mainWnnNer(parameters)


if __name__ == '__main__':
    main()
