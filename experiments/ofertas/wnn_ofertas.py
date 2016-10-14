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
import time
from logging import Formatter
from time import time

import theano.tensor as T
from data.BatchIterator import SyncBatchIterator, AsyncBatchIterator
from data.FeatureGenerator import FeatureGenerator
from data.WordWindowGenerator import WordWindowGenerator

from args.JsonArgParser import JsonArgParser
from data.DatasetReader import DatasetReader
from data.Embedding import EmbeddingFactory, RandomUnknownStrategy, ChosenUnknownStrategy, \
    RandomEmbedding
from data.Lexicon import Lexicon, createLexiconUsingFile, HashLexicon
from model.Model import Model

from model.Objective import NegativeLogLikelihoodOneExample
from model.Prediction import ArgmaxPrediction
from model.SaveModelCallback import ModelWriter, SaveModelCallback
from nnet.ActivationLayer import ActivationLayer, softmax, tanh, sigmoid
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.FlattenLayer import FlattenLayer
from nnet.LinearLayer import LinearLayer
from nnet.MaxPoolingLayer import MaxPoolingLayer
from nnet.WeightGenerator import ZeroWeightGenerator, GlorotUniform, SigmoidGlorot
from optim.Adagrad import Adagrad
from optim.SGD import SGD
from util.jsontools import dict2obj
from model.Metric import LossMetric, AccuracyMetric, FMetric

PARAMETERS = {
    "filters": {"default": ['data.Filters.TransformLowerCaseFilter',
                            'data.Filters.TransformNumberToZeroFilter'],
                "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},
    "train": {"desc": "Training File Path"},
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},
    "lr": {"desc": "learning rate value"},
    "save_model": {"desc": "Path + basename that will be used to save the model (weights and embeddings)."},
    "load_model": {"desc": "Path + basename that will be used to load the model (weights and embeddings)."},
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
    "load_hidden_layer": {"desc": "File containing weights and bias of pre-trained hidden layer."},
    "hidden_activation_function": {"default": "tanh",
                                   "desc": "the activation function of the hidden layer. The possible values are: 'tanh' and 'sigmoid'."},
    "shuffle": {"default": True,
                "desc": "Enable or disable shuffling of the training examples."},
    "normalization": {"desc": "Choose the normalization method to be applied on  word embeddings. " +
                              "The possible values are: 'minmax', 'mean', 'zscore'."},
    "labels": {"desc": "File containing the list of possible labels."},
    "conv_size": {"required": True,
                  "desc": "Size of the convolution layer (number of filters)."},
    "load_method": {"default": "sync",
                    "desc": "Method for loading the training dataset." +
                            "The possible values are: 'sync' and 'async'."},
    "hash_lex_size": {"desc": "Activate the hash lexicon by specifying the hash table size."}
}


class OfertasReader(DatasetReader):
    """
    Lê exemplos de ofertas. O formato o seguinte. Cada linha contém um exemplo (a 
    primeira linha é o cabeçalho). Cada exemplo segue o seguinte formato:
    
    <id_pai> [TAB] <id> [TAB] <desc_norm> [TAB] <categ_shop_desc_nor> [TAB] <price>
    
    onde, <id_pai> é o ID da categoria pai, <id> é o ID da categoria da oferta,
    <desc_norm> é o texto da oferta, <categ_shop_desc_nor> é categoria interna do
    anunciante, e <price> é o preço do produto.
    """

    def __init__(self, filePath):
        """
        :type filePath: String
        :param filePath: dataset path
        """
        self.__filePath = filePath
        self.__log = logging.getLogger(__name__)
        self.__printedNumberTokensRead = False

    def read(self):
        """
        :return: lista de tokens da oferta e sua categoria.
        """
        f = codecs.open(self.__filePath, "r", "utf-8")
        numExs = 0

        # Skip the first line (header).
        f.readline()

        for line in f:
            line = line.strip()

            # Skip blank lines.
            if len(line) == 0:
                continue

            ftrs = [s.strip() for s in line.split('\t')]

            tokens = ftrs[2].split()
            category = ftrs[1]

            numExs += 1

            yield (tokens, category)

        if not self.__printedNumberTokensRead:
            self.__log.info("Number of examples read: %d" % numExs)


class OfertasModelWritter(ModelWriter):
    def __init__(self, savePath, embeddingLayer, linearLayer1, linearLayer2, embedding, lexiconLabel,
                 hiddenActFunction):
        """
        :param savePath: path where the model will be saved

        :type embeddingLayer: nnet.EmbeddingLayer.EmbeddingLayer
        :type linearLayer1: nnet.LinearLayer.LinearLayer
        :type linearLayer2: nnet.LinearLayer.LinearLayer
        :type embedding: data.Embedding.Embedding
        """
        self.__savePath = savePath
        self.__embeddingLayer = embeddingLayer
        self.__linear1 = linearLayer1
        self.__linear2 = linearLayer2
        self.__embedding = embedding
        self.__logging = logging.getLogger(__name__)
        self.__labelLexicon = lexiconLabel
        self.__hiddenActFunction = hiddenActFunction

    def save(self):
        begin = int(time())
        # Saving embedding
        wbFile = codecs.open(self.__savePath + ".wv", "w", encoding="utf-8")
        lexicon = self.__embedding.getLexicon()
        listWords = lexicon.getLexiconList()
        wordEmbeddings = self.__embeddingLayer.getParameters()[0].get_value()

        # wbFile.write(unicode(len(listWords)))
        # wbFile.write(" ")
        # wbFile.write(unicode(self.__embedding.getEmbeddingSize()))
        # wbFile.write("\n")
        #
        # for a in xrange(len(listWords)):
        #     wbFile.write(listWords[a])
        #     wbFile.write(' ')
        #
        #     for i in wordEmbeddings[a]:
        #         wbFile.write(unicode(i))
        #         wbFile.write(' ')
        #
        #     wbFile.write('\n')

        wbFile.close()

        # Savings labels
        param = {
            "labels": self.__labelLexicon.getLexiconList(),
            "hiddenActFunction": self.__hiddenActFunction,
            "unknown": lexicon.getLexicon(lexicon.getUnknownIndex())
        }

        with codecs.open(self.__savePath + ".param", "w", encoding="utf-8") as paramsFile:
            json.dump(param, paramsFile, encoding="utf-8")

        weights = {}

        W1, b1 = self.__linear1.getParameters()
        weights["W_Hidden"] = W1.get_value()
        weights["b_Hidden"] = b1.get_value()

        W2, b2 = self.__linear2.getParameters()

        weights["W_Softmax"] = W2.get_value()
        weights["b_Softmax"] = b2.get_value()

        np.save(self.__savePath, weights)

        self.__logging.info("Model Saved in %d", int(time()) - begin)


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

    loadPath = args.load_model

    if loadPath:
        with codecs.open(loadPath + ".param", "r", encoding="utf-8") as paramsFile:
            param = json.load(paramsFile, encoding="utf-8")

        hiddenActFunctionName = param['hiddenActFunction']
        hiddenActFunction = method_name(hiddenActFunctionName)

        # Loading Embedding
        log.info("Loading Model")
        wordEmbedding = EmbeddingFactory().createFromW2V(loadPath + ".wv", ChosenUnknownStrategy(param["unknown"]))
        labelLexicon = Lexicon()

        for l in param["labels"]:
            labelLexicon.put(l)

        labelLexicon.stopAdd()

        # Loading model
        weights = np.load(loadPath + ".npy").item(0)

        W1 = weights["W_Hidden"]
        b1 = weights["b_Hidden"]
        W2 = weights["W_Softmax"]
        b2 = weights["b_Softmax"]

        hiddenLayerSize = b1.shape[0]
    else:
        W1 = None
        b1 = None
        W2 = None
        b2 = None
        hiddenActFunctionName = args.hidden_activation_function
        hiddenActFunction = method_name(hiddenActFunctionName)

        if args.word_embedding:
            log.info("Reading W2v File")
            wordEmbedding = EmbeddingFactory().createFromW2V(args.word_embedding, RandomUnknownStrategy())
            # TODO: teste
            wordEmbedding.stopAdd()
        elif args.hash_lex_size:
            wordEmbedding = RandomEmbedding(args.word_emb_size,
                                            RandomUnknownStrategy(),
                                            HashLexicon(args.hash_lex_size))
        else:
            wordEmbedding = EmbeddingFactory().createRandomEmbedding(args.word_emb_size)

        # Get the inputs and output
        if args.labels:
            labelLexicon = createLexiconUsingFile(args.labels)
        else:
            labelLexicon = Lexicon()

        if args.load_hidden_layer:
            # Loading Hidden Layer
            log.info("Loading Hidden Layer")

            hl = np.load(args.load_hidden_layer).item(0)

            W1 = hl["W_Encoder"]
            b1 = hl["b_Encoder"]

            hiddenLayerSize = b1.shape[0]

    # Generate word windows.
    wordWindowFeatureGenerator = WordWindowGenerator(wordWindowSize, wordEmbedding, filters, startSymbol, endSymbol)
    # Generate one label per example (list of tokens).
    labelGenerator = TextLabelGenerator(labelLexicon)

    if args.train:
        log.info("Reading training examples")

        trainDatasetReader = OfertasReader(args.train)
        if args.load_method == "sync":
            trainIterator = SyncBatchIterator(trainDatasetReader,
                                              [wordWindowFeatureGenerator],
                                              [labelGenerator],
                                              - 1,
                                              shuffle=shuffle)
        elif args.load_method == "async":
            trainIterator = AsyncBatchIterator(trainDatasetReader,
                                               [wordWindowFeatureGenerator],
                                               [labelGenerator],
                                               - 1,
                                               shuffle=shuffle,
                                               maxqSize=1000)
        else:
            log.error("The argument 'load_method' has an invalid value: %s." % args.load_method)
            sys.exit(1)

        wordEmbedding.stopAdd()
        labelLexicon.stopAdd()

        # Get dev inputs and output
        dev = args.dev
        evalPerIteration = args.eval_per_iteration
        if not dev and evalPerIteration > 0:
            log.error("Argument eval_per_iteration cannot be used without a dev argument.")
            sys.exit(1)

        if dev:
            log.info("Reading development examples")
            devReader = OfertasReader(args.dev)
            devIterator = SyncBatchIterator(devReader,
                                            [wordWindowFeatureGenerator],
                                            [labelGenerator],
                                            - 1,
                                            shuffle=False)
        else:
            devIterator = None
    else:
        trainIterator = None
        devIterator = None

    weightInit = SigmoidGlorot() if hiddenActFunction == sigmoid else GlorotUniform()

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
        log.error("Normalization: unexpected value %s" % normalizeMethod)
        sys.exit(1)

    if normalizeMethod is not None and loadPath is not None:
        log.warn("The word embedding of model was normalized. This can change the result of test.")

    #
    # Build the network model (Theano graph).
    #

    # Matriz de entrada. Cada linha representa um token da oferta. Cada token é
    # representado por uma janela de tokens (token central e alguns tokens
    # próximos). Cada valor desta matriz corresponde a um índice que representa
    # um token no embedding.
    inWords = T.lmatrix("inWords")

    # Categoria correta de uma oferta.
    outLabel = T.lscalar("outLabel")

    # TODO: debug
    # theano.config.compute_test_value = 'warn'
    # ex = trainIterator.next()
    # inWords.tag.test_value = ex[0][0]
    # outLabel.tag.test_value = ex[1][0]

    # Lookup table.
    embeddingLayer = EmbeddingLayer(inWords, wordEmbedding.getEmbeddingMatrix())

    # A saída da lookup table possui 3 dimensões (numTokens, szWindow, szEmbedding).
    # Esta camada dá um flat nas duas últimas dimensões, produzindo uma saída
    # com a forma (numTokens, szWindow * szEmbedding).
    flattenInput = FlattenLayer(embeddingLayer)

    # Convolution layer. Convolução no texto de uma oferta.
    convLinear = LinearLayer(flattenInput,
                             wordWindowSize * wordEmbedding.getEmbeddingSize(),
                             convSize, W=None, b=None,
                             weightInitialization=weightInit)
    maxPooling = MaxPoolingLayer(convLinear)

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

    # Decaimento da taxa de aprendizado.
    decay = 0.0
    if args.decay == "linear":
        decay = 1.0

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

    # Loss function.
    loss = NegativeLogLikelihoodOneExample().calculateError(softmaxAct.getOutput()[0], prediction, outLabel)

    #     if kwargs["lambda"]:
    #         _lambda = kwargs["lambda"]
    #         log.info("Using L2 with lambda= %.2f", _lambda)
    #         loss += _lambda * (T.sum(T.square(hiddenLinear.getParameters()[0])))

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
    model = Model(x=[inWords], y=[outLabel], allLayers=softmaxAct.getLayerSet(), optimizer=opt, prediction=prediction,
                  loss=loss, trainMetrics=trainMetrics, evalMetrics=evalMetrics, testMetrics=testMetrics, mode=mode)

    # Training
    if trainIterator:
        callback = []

        if args.save_model:
            savePath = args.save_model
            modelWriter = OfertasModelWritter(savePath, embeddingLayer,
                                              hiddenLinear, sotmaxLinearInput,
                                              wordEmbedding, labelLexicon,
                                              hiddenActFunctionName)
            callback.append(SaveModelCallback(modelWriter, "eval_acc", True))

        log.info("Training")
        model.train(trainIterator, numEpochs, devIterator, evalPerIteration=evalPerIteration, callbacks=callback)

    # Testing
    if args.test:
        log.info("Reading test examples")
        testReader = OfertasReader(args.test)
        testIterator = SyncBatchIterator(testReader,
                                         [wordWindowFeatureGenerator],
                                         [labelGenerator],
                                         - 1,
                                         shuffle=False)

        log.info("Testing")
        model.test(testIterator)


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

    logging.config.fileConfig(os.path.join(path, 'logging.conf'), defaults={})

    argsDict = JsonArgParser(PARAMETERS).parse(sys.argv[1])
    args = dict2obj(argsDict, 'OfertaArguments')
    logging.getLogger(__name__).info(argsDict)

    main(args)
