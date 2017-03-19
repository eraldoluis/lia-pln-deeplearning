#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import importlib
import json
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
from data.DatasetReader import DatasetReader
from data.FeatureGenerator import FeatureGenerator
from data.Lexicon import Lexicon
from data.WordWindowGenerator import WordWindowGenerator
from model.Metric import LossMetric, AccuracyMetric, FMetric, PredictedProbabilities
from model.Model import Model
from model.Objective import NegativeLogLikelihoodOneExample
from model.Prediction import ArgmaxPrediction
from nnet.ActivationLayer import ActivationLayer, softmax, tanh, sigmoid
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.FlattenLayer import FlattenLayer
from nnet.LinearLayer import LinearLayer
from nnet.MaxPoolingLayer import MaxPoolingLayer
from nnet.WeightGenerator import ZeroWeightGenerator, GlorotUniform, SigmoidGlorot
from optim.Adagrad import Adagrad
from optim.SGD import SGD
from util.jsontools import dict2obj
from data.Embedding import Embedding
from model.BasicModel import BasicModel

PARAMETERS = {
    "filters": {"default": ['data.Filters.TransformLowerCaseFilter',
                            'data.Filters.TransformNumberToZeroFilter'],
                "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},
    "train": {"desc": "Training File Path"},
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},
    "lr": {"desc": "learning rate value"},
    "test": {"desc": "Test set file path"},
    "dev": {"desc": "Development set file path"},
    "eval_per_iteration": {"default": 0,
                           "desc": "Eval model after this number of iterations."},
    "hidden_size": {"default": 300,
                    "desc": "The number of neurons in the hidden layer"},
    "word_window_size": {"default": 5,
                         "desc": "The size of words for the wordsWindow"},
    "word_emb_size": {"default": 100,
                      "desc": "size of word embedding"},
    "word_embedding": {"desc": "word embedding File Path"},
    "start_symbol": {"default": "</s>",
                     "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>",
                   "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc": "Random number generator seed."},
    "alg": {"default": "sgd",
            "desc": "Optimization algorithm to be used. Options are: 'sgd', 'adagrad'."},
    "decay": {"default": "linear",
              "desc": "Set the learning rate update strategy. Options are: 'none' and 'linear'."},
    "shuffle": {"default": True,
                "desc": "Enable or disable shuffling of the training examples."},
    "wv_normalization": {"desc": "Choose the normalization method to be applied on  word embeddings. " +
                                 "The possible values are: 'minmax', 'mean', 'zscore'."},
    "labels": {"desc": "File containing the list of possible labels."},
    "conv_size": {"required": True,
                  "desc": "Size of the convolution layer (number of filters)."},
    "fix_word_embedding": {
        "desc": "Fix the word embedding (do not update it during training).",
        "default": False
    }
}


class DocReader(DatasetReader):
    """
    Lê exemplos de documentos de acordo com o formato a seguir.
    Cada linha contém um exemplo.
    Cada exemplo segue o seguinte formato:
    
    <category> [TAB] <text>
    """

    def __init__(self, filePath, header=False):
        """
        :type filePath: String
        :param filePath: dataset path

        :type header: bool
        :param header: whether the given file include a header in the first line
        """
        self.__filePath = filePath
        self.__log = logging.getLogger(__name__)
        self.__printedNumberTokensRead = False
        self.__header = header

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__f:
            self.__f.close()
            self.__f = None

    def read(self):
        """
        :return: lista de tokens da oferta e sua categoria.
        """
        f = codecs.open(self.__filePath, "r", "utf-8")
        self.__f = f
        numExs = 0

        # Skip the first line (header).
        if self.__header:
            f.readline()

        for line in f:
            line = line.strip()

            # Skip blank lines.
            if len(line) == 0:
                continue

            (category, text) = [s.strip() for s in line.split('\t')]

            numExs += 1

            yield (text, category)

        if not self.__printedNumberTokensRead:
            self.__log.info("Number of examples read: %d" % numExs)


class TextLabelGenerator(FeatureGenerator):
    """
    Generates one label per example (in general, the input is a piece of text).
    This label generator is usually used for document (text) classification.
    """

    def __init__(self, labelLexicon):
        """
        :type labelLexicon: data.Lexicon.Lexicon
        :param labelLexicon:
        """
        self.__labelLexicon = labelLexicon

    def generate(self, label):
        """
        Return the code for the given label.

        :type labels: list[basestring]
        :param labels:

        :return: li
        """

        y = self.__labelLexicon.put(label)

        if y == -1:
            raise Exception("Label doesn't exist: %s" % label)

        return y


def main(args):
    log = logging.getLogger(__name__)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    lr = args.lr
    startSymbol = args.start_symbol
    endSymbol = args.end_symbol
    numEpochs = args.num_epochs
    shuffle = args.shuffle
    normalizeMethod = args.wv_normalization
    wordWindowSize = args.word_window_size
    hiddenLayerSize = args.hidden_size
    convSize = args.conv_size

    # Load classes for filters.
    filters = []
    for filterName in args.filters:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Filtro: " + moduleName + " " + className)
        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    W1 = None
    b1 = None
    W2 = None
    b2 = None
    hiddenActFunction = tanh

    if args.word_embedding:
        log.info("Reading W2v File")
        (lexicon, wordEmbedding) = Embedding.fromWord2Vec(args.word_embedding, unknownSymbol='unknown')
        lexicon.stopAdd()
    else:
        wordEmbedding = EmbeddingFactory().createRandomEmbedding(args.word_emb_size)

    # Get the inputs and output
    if args.labels:
        labelLexicon = Lexicon.fromTextFile(args.labels, hasUnknowSymbol=False)
    else:
        labelLexicon = Lexicon()

    #
    # Build the network model (Theano graph).
    #

    # TODO: debug
    # theano.config.compute_test_value = 'warn'
    # ex = trainIterator.next()
    # inWords.tag.test_value = ex[0][0]
    # outLabel.tag.test_value = ex[1][0]

    # Matriz de entrada. Cada linha representa um token da oferta. Cada token é
    # representado por uma janela de tokens (token central e alguns tokens
    # próximos). Cada valor desta matriz corresponde a um índice que representa
    # um token no embedding.
    inWords = T.lmatrix("inWords")

    # Categoria correta de uma oferta.
    outLabel = T.lscalar("outLabel")

    # List of input tensors. One for each input layer.
    inputTensors = [inWords]

    # Whether the word embedding will be updated during training.
    embLayerTrainable = not args.fix_word_embedding

    if not embLayerTrainable:
        log.info("Not updating the word embedding!")

    # Lookup table for word features.
    embeddingLayer = EmbeddingLayer(inWords, wordEmbedding.getEmbeddingMatrix(), trainable=embLayerTrainable)

    # A saída da lookup table possui 3 dimensões (numTokens, szWindow, szEmbedding).
    # Esta camada dá um flat nas duas últimas dimensões, produzindo uma saída
    # com a forma (numTokens, szWindow * szEmbedding).
    flattenInput = FlattenLayer(embeddingLayer)

    # Random weight initialization procedure.
    weightInit = SigmoidGlorot() if hiddenActFunction == sigmoid else GlorotUniform()

    # Convolution layer. Convolução no texto de um documento.
    convLinear = LinearLayer(flattenInput,
                             wordWindowSize * wordEmbedding.getEmbeddingSize(),
                             convSize, W=None, b=None,
                             weightInitialization=weightInit)

    # Max pooling layer.
    maxPooling = MaxPoolingLayer(convLinear)

    # Generate word windows.
    wordWindowFeatureGenerator = WordWindowGenerator(wordWindowSize, lexicon, filters, startSymbol, endSymbol)

    # List of input generators.
    inputGenerators = [wordWindowFeatureGenerator]

    # Hidden layer.
    hiddenLinear = LinearLayer(maxPooling,
                               convSize,
                               hiddenLayerSize,
                               W=W1, b=b1,
                               weightInitialization=weightInit)
    hiddenAct = ActivationLayer(hiddenLinear, hiddenActFunction)

    # Entrada linear da camada softmax.
    sotmaxLinearInput = LinearLayer(hiddenAct,
                                    hiddenLayerSize,
                                    labelLexicon.getLen(),
                                    W=W2, b=b2,
                                    weightInitialization=ZeroWeightGenerator())
    # Softmax.
    # softmaxAct = ReshapeLayer(ActivationLayer(sotmaxLinearInput, softmax), (1, -1))
    softmaxAct = ActivationLayer(sotmaxLinearInput, softmax)

    # Prediction layer (argmax).
    prediction = ArgmaxPrediction(None).predict(softmaxAct.getOutput())

    # Loss function.
    loss = NegativeLogLikelihoodOneExample().calculateError(softmaxAct.getOutput()[0], prediction, outLabel)

    # Output generator: generate one label per offer.
    outputGenerators = [TextLabelGenerator(labelLexicon)]

    if args.train:
        trainDatasetReader = DocReader(args.train)

        log.info("Reading training examples...")
        trainIterator = SyncBatchIterator(trainDatasetReader,
                                          inputGenerators,
                                          outputGenerators,
                                          -1,
                                          shuffle=shuffle)
        lexicon.stopAdd()
        labelLexicon.stopAdd()

        # Get dev inputs and output
        dev = args.dev
        evalPerIteration = args.eval_per_iteration
        if not dev and evalPerIteration > 0:
            log.error("Argument eval_per_iteration cannot be used without a dev argument.")
            sys.exit(1)

        if dev:
            log.info("Reading development examples")
            devReader = DocReader(args.dev)
            devIterator = SyncBatchIterator(devReader,
                                            inputGenerators,
                                            outputGenerators,
                                            -1,
                                            shuffle=False)
        else:
            devIterator = None
    else:
        trainIterator = None
        devIterator = None

    if normalizeMethod == "minmax":
        log.info("Normalization: minmax")
        wordEmbedding.minMaxNormalization()
    elif normalizeMethod == "mean":
        log.info("Normalization: mean normalization")
        wordEmbedding.meanNormalization()
    elif normalizeMethod == "zscore":
        log.info("Normalization: zscore normalization")
        wordEmbedding.zscoreNormalization()
    elif normalizeMethod:
        log.error("Normalization: unknown value %s" % normalizeMethod)
        sys.exit(1)

    # Decaimento da taxa de aprendizado.
    if args.decay == "linear":
        decay = 1.0
    elif args.decay == "none":
        decay = 0.0
    else:
        log.error("Unknown decay strategy %s. Expected: none or linear." % args.decay)
        sys.exit(1)

    # Algoritmo de aprendizado.
    if args.alg == "adagrad":
        log.info("Using Adagrad")
        opt = Adagrad(lr=lr, decay=decay)
    elif args.alg == "sgd":
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)
    else:
        log.error("Unknown algorithm: %s. Expected values are: adagrad or sgd." % args.alg)
        sys.exit(1)

    # TODO: debug
    # opt.lr.tag.test_value = 0.05

    # Printing embedding information.
    dictionarySize = wordEmbedding.getNumberOfVectors()
    embeddingSize = wordEmbedding.getEmbeddingSize()
    log.info("Dictionary size: %d" % dictionarySize)
    log.info("Embedding size: %d" % embeddingSize)
    log.info("Number of categories: %d" % labelLexicon.getLen())

    # Train metrics.
    trainMetrics = None
    if trainIterator:
        trainMetrics = [
            LossMetric("TrainLoss", loss),
            AccuracyMetric("TrainAccuracy", outLabel, prediction)
        ]

    # Evaluation metrics.
    evalMetrics = None
    if devIterator:
        evalMetrics = [
            LossMetric("EvalLoss", loss),
            AccuracyMetric("EvalAccuracy", outLabel, prediction)
        ]

    # Test metrics.
    testMetrics = None
    if args.test:
        testMetrics = [
            LossMetric("TestLoss", loss),
            AccuracyMetric("TestAccuracy", outLabel, prediction)
        ]

    # TODO: debug
    # mode = theano.compile.debugmode.DebugMode(optimizer=None)
    mode = None
    model = BasicModel(x=inputTensors, y=[outLabel], allLayers=softmaxAct.getLayerSet(), optimizer=opt,
                       prediction=prediction, loss=loss, trainMetrics=trainMetrics, evalMetrics=evalMetrics,
                       testMetrics=testMetrics, mode=mode)

    # Training
    if trainIterator:
        log.info("Training")
        model.train(trainIterator, numEpochs, devIterator, evalPerIteration=evalPerIteration)

    # Testing
    if args.test:
        log.info("Reading test examples")
        testReader = DocReader(args.test)
        testIterator = SyncBatchIterator(testReader,
                                         inputGenerators,
                                         outputGenerators,
                                         -1,
                                         shuffle=False)

        log.info("Testing")
        model.test(testIterator)


if __name__ == '__main__':
    # Load logging configuration.
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    logging.config.fileConfig(os.path.join(path, 'logging.conf'), defaults={})

    log = logging.getLogger(__name__)

    if len(sys.argv) != 2:
        log.error('Syntax error! Expected JSON arguments file.')
        sys.exit(1)

    # Load arguments from JSON input file.
    argsDict = JsonArgParser(PARAMETERS).parse(sys.argv[1])
    args = dict2obj(argsDict, 'DocClassificationArguments')
    logging.getLogger(__name__).info(argsDict)

    main(args)
