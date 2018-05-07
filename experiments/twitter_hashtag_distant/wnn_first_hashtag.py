#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import importlib
import logging
import logging.config
import numpy as np
import os
import random
import sys

from theano import tensor
from theano.scalar import float64

from args.JsonArgParser import JsonArgParser
from data.BatchIterator import SyncBatchIterator, AsyncBatchIterator
from data.DatasetReader import DatasetReader
from data.Embedding import Embedding
from data.FeatureGenerator import FeatureGenerator
from data.Lexicon import Lexicon
from data.WordWindowGenerator import WordWindowGenerator
from model.BasicModel import BasicModel
from model.Metric import LossMetric, AccuracyMetric, FMetric, PredictedProbabilities
from model.Objective import NegativeLogLikelihoodOneExample
from model.Prediction import ArgmaxPrediction
from nnet.ActivationLayer import ActivationLayer, softmax, tanh
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.FlattenLayer import FlattenLayer
from nnet.LinearLayer import LinearLayer
from nnet.MaxPoolingLayer import MaxPoolingLayer
from nnet.WeightGenerator import ZeroWeightGenerator, GlorotUniform
from optim.Adagrad import Adagrad
from optim.SGD import SGD
from util.jsontools import dict2obj

PARAMETERS = {
    "filters": {"default": ['data.Filters.TransformLowerCaseFilter',
                            'data.Filters.TransformNumberToZeroFilter'],
                "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},
    "label_lexicon": {"desc": "File containing the list of labels. " +
                              "The index used for each label will be given by the order in this file."},
    "labels": {"desc": "List of labels." +
                       "The index used for each label will be given by the order in this list."},
    # Datasets.
    "train": {"desc": "Training File Path"},
    "load_method": {"default": "sync",
                    "desc": "Method for loading the training dataset (sync or async)."},
    "test": {"desc": "Test set file path"},
    "dev": {"desc": "Development set file path"},
    "eval_per_iteration": {"default": 0,
                           "desc": "Eval model after this number of iterations."},
    # Layer: input (word embedding).
    "word_window_size": {"default": 5,
                         "desc": "The size of words for the wordsWindow"},
    "word_lexicon": {"desc": "word embedding File Path"},
    "word_emb_size": {"default": 100,
                      "desc": "size of word embedding"},
    "word_embedding": {"desc": "word embedding File Path"},
    "normalization": {"desc": "Choose the normalization method to be applied on  word embeddings. " +
                              "The possible values are: 'minmax', 'mean', 'zscore'."},
    "fix_word_embedding": {
        "default": False,
        "desc": "Fix the word embedding (do not update it during training)."},
    "start_symbol": {"default": "</s>",
                     "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>",
                   "desc": "Object that will be place when the end limit of list is exceeded"},
    # Layer: convolution.
    "conv_size": {"required": True,
                  "desc": "Size of the convolution layer (number of filters)."},
    "conv_act": {"default": False,
               "desc": "Whether to use an activation function after the convolution layer."},
    # Layer: hidden.
    "hidden_size": {"default": 300,
                    "desc": "The number of neurons in the hidden layer"},
    "hidden": {"default": True,
               "desc": "Whether to use a hidden layer after the max pooling or not."},

    # Learning algorithm.
    "alg": {"default": "sgd",
            "desc": "Optimization algorithm to be used. Options are: 'sgd', 'adagrad'."},
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},
    "lr": {"desc": "learning rate value"},
    "decay": {"default": "linear",
              "desc": "Set the learning rate update strategy (none or linear)."},
    "shuffle": {"default": True,
                "desc": "Enable or disable shuffling of the training examples."},
    "label_weights": {"desc": "List of weights for each label. These weights are used in the loss function."},
    "seed": {"desc": "Random number generator seed."},

    # Load pre trained model.
    "load_conv": {"desc": "pre trained convolution layer File Path"},
    "load_hiddenLayer": {"desc": "pre trained hidden layer File Path"},
    "load_softmax": {"desc": "pre trained softmax layer File Path"},

    # Save model after train.
    "save_wordEmbedding": {"desc": "save trained word embedding to File Path"},
    "save_conv": {"desc": "save trained convolution layer to File Path"},
    "save_hiddenLayer": {"desc": "save trained hidden layer to File Path"},
    "save_softmax": {"desc": "save trained softmax to File Path"}
}


class ShortDocReader(DatasetReader):
    """
    Lê exemplos de documentos curtos. O formato é o seguinte. Cada linha contém um documento.
    Um documento é composto por um texto e uma classe, separados por um caractere <TAB>.
    """

    def __init__(self, filePath):
        """
        :type filePath: String
        :param filePath: dataset path
        """
        self.__f = None
        self.__filePath = filePath
        self.__log = logging.getLogger(__name__)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__f is not None:
            self.__f.close()
            self.__f = None

    def read(self):
        """
        :return: lista de tokens da oferta e sua categoria.
        """
        f = codecs.open(self.__filePath, "r", "utf-8")
        self.__f = f
        numExs = 0

        for line in f:
            # Skip blank lines.
            if len(line) == 0:
                continue

            (text, cls) = line.split('\t')

            text = text.strip()
            cls = cls.strip()

            numExs += 1

            yield (text, cls)


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

        :type label: list[basestring]
        :param label:

        :return: li
        """

        # This has been changed to it gets only the first class/label
        y = self.__labelLexicon.put(label.split(' ')[0])

        if y == -1:
            raise Exception("Label doesn't exist: %s" % label)

        return y


def main():
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    logging.config.fileConfig(os.path.join(path, 'logging.conf'), defaults={})
    log = logging.getLogger(__name__)

    if len(sys.argv) != 2:
        log.error("Missing argument: <JSON config file>")
        exit(1)

    argsDict = JsonArgParser(PARAMETERS).parse(sys.argv[1])
    args = dict2obj(argsDict, 'ShortDocArguments')
    logging.getLogger(__name__).info(argsDict)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    lr = args.lr
    startSymbol = args.start_symbol
    endSymbol = args.end_symbol
    numEpochs = args.num_epochs
    shuffle = args.shuffle
    normalizeMethod = args.normalization
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

    wordEmbedding = None
    if args.word_embedding:
        log.info("Reading W2v File")
        (wordLexicon, wordEmbedding) = Embedding.fromWord2Vec(args.word_embedding, unknownSymbol="__UNKNOWN__")
        wordLexicon.stopAdd()
    elif args.word_lexicon and args.word_emb_size:
        wordLexicon = Lexicon.fromTextFile(args.word_lexicon, hasUnknowSymbol=False)
        wordEmbedding = Embedding(wordLexicon, embeddingSize=args.word_emb_size)
        wordLexicon.stopAdd()
    else:
        log.error("You must provide argument word_embedding or word_lexicon and word_emb_size")

    # Create the lexicon of labels.
    labelLexicon = None
    if args.labels is not None:
        if args.label_lexicon is not None:
            log.error("Only one of the parameters label_lexicon and labels can be provided!")
            exit(1)
        labelLexicon = Lexicon.fromList(args.labels, hasUnknowSymbol=False)
    elif args.label_lexicon is not None:
        labelLexicon = Lexicon.fromTextFile(args.label_lexicon, hasUnknowSymbol=False)
    else:
        log.error("One of the parameters label_lexicon or labels must be provided!")
        exit(1)

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
    inWords = tensor.lmatrix("inWords")

    # Categoria correta de uma oferta.
    outLabel = tensor.lscalar("outLabel")

    # List of input tensors. One for each input layer.
    inputTensors = [inWords]

    # Whether the word embedding will be updated during training.
    embLayerTrainable = not args.fix_word_embedding

    if not embLayerTrainable:
        log.info("Not updating the word embedding!")

    # Lookup table for word features.
    embeddingLayer = EmbeddingLayer(inWords, wordEmbedding.getEmbeddingMatrix(), trainable=embLayerTrainable)

    # if not args.train and args.load_wordEmbedding:
    #     attrs = np.load(args.load_wordEmbedding)
    #     embeddingLayer.load(attrs)
    #     log.info("Loaded word embedding (shape %s) from file %s" % (
    #         str(attrs[0].shape), args.load_wordEmbedding))

    # A saída da lookup table possui 3 dimensões (numTokens, szWindow, szEmbedding).
    # Esta camada dá um flat nas duas últimas dimensões, produzindo uma saída
    # com a forma (numTokens, szWindow * szEmbedding).
    flattenInput = FlattenLayer(embeddingLayer)

    # Random weight initialization procedure.
    weightInit = GlorotUniform()

    # Convolution layer. Convolução no texto de uma oferta.
    convW = None
    convb = None

    convLinear = LinearLayer(flattenInput,
                             wordWindowSize * wordEmbedding.getEmbeddingSize(),
                             convSize, W=convW, b=convb,
                             weightInitialization=weightInit)

    # TODO Igor, verificar
    if args.load_conv:
        convLinear.load(args.load_conv)
        # convNPY = np.load(args.load_conv)
        # convW = convNPY[0]
        # convb = convNPY[1]
        (W, b) = convLinear.getParameters()
        log.info("Loaded convolutional layer (shapes W=%s B=%s) from file %s" % (str(W.shape), str(b.shape), args.load_conv))

    if args.conv_act:
        convOut = ActivationLayer(convLinear, tanh)
    else:
        convOut = convLinear

    # Max pooling layer.
    maxPooling = MaxPoolingLayer(convOut)

    softmaxInput = None
    softmaxInputLen = -1
    if args.hidden:
        # Hidden layer.
        if not args.train and args.load_hiddenLayer:
            hiddenNPY = np.load(args.load_hiddenLayer)
            W1 = hiddenNPY[0]
            b1 = hiddenNPY[1]
            log.info("Loaded hidden layer (shape %s) from file %s" % (str(W1.shape), args.load_hiddenLayer))

        hiddenLinear = LinearLayer(maxPooling,
                                   convSize,
                                   hiddenLayerSize,
                                   W=W1, b=b1,
                                   weightInitialization=weightInit)

        hiddenAct = ActivationLayer(hiddenLinear, tanh)

        # Entrada linear da camada softmax.
        if not args.train and args.load_softmax:
            hiddenNPY = np.load(args.load_softmax)
            W2 = hiddenNPY[0]
            b2 = hiddenNPY[1]
            log.info("Loaded softmax layer (shape %s) from file %s" % (str(W2.shape), args.load_softmax))

        softmaxInput = hiddenAct
        softmaxInputLen = hiddenLayerSize
    else:
        softmaxInput = maxPooling
        softmaxInputLen = convSize

    sotmaxLinearInput = LinearLayer(softmaxInput,
                                    softmaxInputLen,
                                    labelLexicon.getLen(),
                                    W=W2, b=b2,
                                    weightInitialization=ZeroWeightGenerator())

    # Softmax.
    # softmaxAct = ReshapeLayer(ActivationLayer(sotmaxLinearInput, softmax), (1, -1))
    softmaxAct = ActivationLayer(sotmaxLinearInput, softmax)

    # Prediction layer (argmax).
    prediction = ArgmaxPrediction(None).predict(softmaxAct.getOutput())

    # Loss function.
    if args.label_weights is not None and len(args.label_weights) != labelLexicon.getLen():
        log.error("Number of label weights (%d) is different from number of labels (%d)!" % (
            len(args.label_weights), labelLexicon.getLen()))
    nlloe = NegativeLogLikelihoodOneExample(weights=args.label_weights)
    loss = nlloe.calculateError(softmaxAct.getOutput()[0], prediction, outLabel)

    # Input generators: word window.
    inputGenerators = [WordWindowGenerator(wordWindowSize, wordLexicon, filters, startSymbol, endSymbol)]

    # Output generator: generate one label per offer.
    outputGenerators = [TextLabelGenerator(labelLexicon)]
    # outputGenerators = [lambda label: labelLexicon.put(label)]

    evalPerIteration = None
    if args.train:
        trainDatasetReader = ShortDocReader(args.train)
        if args.load_method == "sync":
            log.info("Reading training examples...")
            trainIterator = SyncBatchIterator(trainDatasetReader,
                                              inputGenerators,
                                              outputGenerators,
                                              - 1,
                                              shuffle=shuffle)
            wordLexicon.stopAdd()
        elif args.load_method == "async":
            log.info("Examples will be asynchronously loaded.")
            trainIterator = AsyncBatchIterator(trainDatasetReader,
                                               inputGenerators,
                                               outputGenerators,
                                               - 1,
                                               shuffle=shuffle,
                                               maxqSize=1000)
        else:
            log.error("The argument 'load_method' has an invalid value: %s." % args.load_method)
            sys.exit(1)

        labelLexicon.stopAdd()

        # Get dev inputs and output
        dev = args.dev
        evalPerIteration = args.eval_per_iteration
        if not dev and evalPerIteration > 0:
            log.error("Argument eval_per_iteration cannot be used without a dev argument.")
            sys.exit(1)

        if dev:
            log.info("Reading development examples")
            devReader = ShortDocReader(args.dev)
            devIterator = SyncBatchIterator(devReader,
                                            inputGenerators,
                                            outputGenerators,
                                            - 1,
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
    decay = None
    if args.decay == "none":
        decay = 0.0
    elif args.decay == "linear":
        decay = 1.0
    else:
        log.error("Unknown decay parameter %s." % args.decay)
        exit(1)

    # Algoritmo de aprendizado.
    if args.alg == "adagrad":
        log.info("Using Adagrad")
        opt = Adagrad(lr=lr, decay=decay)
    elif args.alg == "sgd":
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)
    else:
        log.error("Unknown algorithm: %s." % args.alg)
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
            AccuracyMetric("EvalAccuracy", outLabel, prediction),
            FMetric("EvalFMetric", outLabel, prediction, labels=labelLexicon.getLexiconDict().values())
        ]

    # Test metrics.
    testMetrics = None
    if args.test:
        testMetrics = [
            LossMetric("TestLoss", loss),
            AccuracyMetric("TestAccuracy", outLabel, prediction),
            FMetric("TestFMetric", outLabel, prediction, labels=labelLexicon.getLexiconDict().values())
        ]

    # TODO: debug
    # mode = theano.compile.debugmode.DebugMode(optimizer=None)
    mode = None
    model = BasicModel(x=inputTensors,
                       y=[outLabel],
                       allLayers=softmaxAct.getLayerSet(),
                       optimizer=opt,
                       prediction=prediction,
                       loss=loss,
                       trainMetrics=trainMetrics,
                       evalMetrics=evalMetrics,
                       testMetrics=testMetrics,
                       mode=mode)

    # Training
    if trainIterator:
        log.info("Training")
        model.train(trainIterator, numEpochs, devIterator, evalPerIteration=evalPerIteration)

    # Saving model after training
        if args.save_wordEmbedding:
            embeddingLayer.saveAsW2V(args.save_wordEmbedding, lexicon=wordLexicon)
            log.info("Saved word to vector to file: %s" % (args.save_wordEmbedding))
        if args.save_conv:
            convLinear.save(args.save_conv)
            log.info("Saved convolution layer to file: %s" % (args.save_conv))
        if args.save_hiddenLayer:
            hiddenLinear.save(args.save_hiddenLayer)
            log.info("Saved hidden layer to file: %s" % (args.save_hiddenLayer))
        if args.save_softmax:
            sotmaxLinearInput.save(args.save_softmax)
            log.info("Saved softmax to file: %s" % (args.save_softmax))

    # Testing
    if args.test:
        log.info("Reading test examples")
        testReader = ShortDocReader(args.test)
        testIterator = SyncBatchIterator(testReader,
                                         inputGenerators,
                                         outputGenerators,
                                         - 1,
                                         shuffle=False)

        log.info("Testing")
        model.test(testIterator)


if __name__ == '__main__':
    main()
