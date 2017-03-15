#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script trains in a unsupervised way the word embeddings and hidden layer of wnn
using negative sampling. Different of the word2vec, this script has the objective to train a classifier
that will be able to predict if word window is feasible or not. The only difference between
the  architecture of this classifier and wnn is the output layer, which is a softmax regression and not
a normal softmax.
To create a noise window word, we change the word in the middle of the feasible window
by one word that is sampled based in a unigram distribution, As Mikolov, calculate unigram probability
of a word as p(word) = p(word)^(3/4)/Z, where p(word) is the count(word)/|size_corpus| and Z is a
normalization constant. For each feasible window, we create k noise windows.
"""
import importlib
import logging.config
import os
import random
import sys

import numpy as np
import theano
import theano.tensor as T

from args.JsonArgParser import JsonArgParser
from data.BatchIterator import SyncBatchIterator
from data.ConstantLabel import ConstantLabel
from data.Embedding import Embedding
from data.LabelGenerator import LabelGenerator
from data.Lexicon import Lexicon
from data.TokenDatasetReader import TokenReader, TokenLabelReader
from data.WordWindowGenerator import WordWindowGenerator
from model.Callback import Callback
from model.Metric import LossMetric, ActivationMetric, DerivativeMetric, AccuracyMetric
from model.ModelWriter import ModelWriter
from model.NegativeSamplingModel import NegativeSamplingModel
from model.NegativeSamplingModelWithSupervised import NegativeSamplingModelWithSupervised
from model.Objective import NegativeLogLikelihood
from model.Prediction import ArgmaxPrediction
from nnet.ActivationLayer import ActivationLayer, tanh, sigmoid, softmax
from nnet.DotLayer import DotLayer
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.FlattenLayer import FlattenLayer
from nnet.LinearLayer import LinearLayer
from nnet.ReshapeLayer import ReshapeLayer
from nnet.WeightGenerator import ZeroWeightGenerator
from optim.Adagrad import Adagrad
from optim.SGD import SGD
from util.Sampler import Sampler
from util.jsontools import dict2obj
from util.util import getFilters

PARAMETERS = {
    # Required
    "train_unsupervised": {"required": True, "desc": "Use text data from file to train the model",},
    "train_supervised": {"required": True, "desc": "",},

    "token_label_separator": {"required": True,
                              "desc": "specify the character that is being used to separate the token from the label in the dataset."},

    "lambda_loss": {"required": True, "desc": "",},
    "label_file": {"required": True, "desc": "file with all possible labels"},
    "word_filters": {"required": True, "desc": "",},

    "lr": {"required": True, "desc": "Set the starting learning rate"},
    "hidden_size": {"required": True, "desc": "size of the hidden layer"},
    "num_epochs": {"required": True,
                   "desc": "Number of epochs: how many iterations over the training set."},
    "noise_rate": {"required": True, "desc": "Number of noise examples",},

    # Word embedding options
    "word_embedding_size": {"required": True,
                            "desc": "the size of the word embedding. This parameter will be used when word_embedding is None."},

    # Model options
    "window_size": {"default": 5, "desc": "size of the window size"},

    # Trainig options
    "min_lr": {"default": 0.0001, "desc": "this is the miminum value which the lr can have"},
    "num_examples_updt_lr": {"default": 10000, "desc": "The lr update is done for each numExUpdLr examples read"},
    # "batch_size": {"default": 1},
    "t": {"default": 10 ** -5,
          "desc": "Set threshold for occurrence of words. Those that appear with higher frequency in the training data "
                  "will be randomly down-sampled; default is 1e-5, useful range is (0, 1e-10)"},

    # Vocabulary and frequency options
    "power": {"default": 0.75, "desc": "q(w)^power, where q(w) is the unigram distribution."},
    "min_count": {"default": 5,
                  "desc": "This will discard words that appear less than n times",},

    # Other options
    "shuffle": {"default": True, "desc": "able or disable the shuffle of training examples."},
    "save_model": {"desc": "Path + basename that will be used to save the weights and embeddings to be saved."},

    "start_symbol": {"default": "</s>",
                     "desc": "Object that will be place when the initial limit of list is exceeded"},
    # "end_symbol": {"default": "</s>",
    #                "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc": ""},
    "enable_activation_statistics": {"desc": "Enable to track the activations value of the main layers",
                                     "default": False},
    "enable_derivative_statistics": {
        "desc": "Enable to track the derivative of some parameters and activation functions. ", "default": False},
    "decay": {"default": "WORD2VEC_DEFAULT",
              "desc": "Set the learning rate update strategy. WORD2VEC_DEFAULT and DIVIDE_EPOCH are the options available"},
    "adagrad": {"desc": "Activate AdaGrad updates.", "default": False}
}


class MetricCB(Callback):
    def __init__(self, metric, evalPerIteration):
        super(MetricCB, self).__init__()

        self.__metric = metric
        self.__evalPerIteration = evalPerIteration
        self.__logger = logging.getLogger(__name__)
        self.__numberIteration = 0

    def onBatchEnd(self, batch, logs={}):
        self.__numberIteration += 1

        if self.__numberIteration % self.__evalPerIteration == 0:
            self.__logger.info({
                "type": "metric",
                "subtype": "train",
                "iteration": self.__numberIteration,
                "name": self.__metric.getName(),
                "values": self.__metric.getValues()
            })

            self.__metric.reset()



def mainWnnNegativeSampling(args):
    # Reading parameters
    embeddingMatrix = None
    wordEmbeddingSize = args.word_embedding_size
    windowSize = args.window_size
    hiddenLayerSize = args.hidden_size
    startSymbol = args.start_symbol
    # endSymbol = args.end_symbol
    endSymbol = startSymbol
    noiseRate = args.noise_rate
    labelLexicon = Lexicon.fromTextFile(args.label_file, False, lexiconName="label_lexicon")

    # todo: o algoritmo não suporta mini batch. Somente treinamento estocástico.
    batchSize = 1

    shuffle = args.shuffle
    lr = args.lr
    numEpochs = args.num_epochs
    power = args.power

    minLr = args.min_lr
    numExUpdLr = args.num_examples_updt_lr

    log = logging.getLogger(__name__)

    log.info(str(args))

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    parametersToSaveOrLoad = {"hidden_size", "window_size", "start_symbol"}

    # Calculate the frequency of each word
    trainUnsupervisedReader = TokenReader(args.train_unsupervised)
    wordLexicon = Lexicon("UUKNNN", "lexicon")
    wordLexicon.put(startSymbol, False)

    totalNumOfTokens = 0
    for tokens, labels in trainUnsupervisedReader.read():
        # we don't count the </s>, because this token is only insert in the sentence to count its frequency.
        totalNumOfTokens += len(tokens)

        # Word2vec considers that the number of lines is the frequency of </s>
        tokens += [startSymbol]

        for token in tokens:
            wordLexicon.put(token)

    # Prune the words with the frequency less than min_count
    wordLexicon.prune(args.min_count)
    wordLexicon.stopAdd()

    # Calculte the unigram distribution
    frequency = np.power(wordLexicon.getFrequencyOfAllWords(), power)
    total = float(frequency.sum())

    # # Print the distribution of all words
    # for _ in xrange(len(frequency)):
    #     print "%s\t%d\t%.4f" % (wordLexicon.getLexicon(_), frequency[_],frequency[_]/float(total))

    sampler = Sampler(frequency / float(total))

    # Create a random embedding for each word
    inputWordEmbedding = Embedding(wordLexicon, None, wordEmbeddingSize)
    outputWordEmbedding = Embedding(wordLexicon, None, hiddenLayerSize)
    # biasWordEmbedding = Embedding(wordLexicon, None, 1)

    log.info("Lexicon size: %d" % (wordLexicon.getLen()))

    # Create NN
    xUns = T.lmatrix("word_window")
    words = T.lvector("correct_noise_words")
    yUns = T.lvector("labels")

    inputWordEmbeddingLayer = EmbeddingLayer(xUns, inputWordEmbedding.getEmbeddingMatrix(), name="embedding")
    flatten = FlattenLayer(inputWordEmbeddingLayer)

    linear1 = LinearLayer(flatten, wordEmbeddingSize * windowSize, hiddenLayerSize, name="linear1")
    act1 = ActivationLayer(linear1, tanh)

    # A logistic regression
    outputWordEmbeddingLayer = EmbeddingLayer(words, outputWordEmbedding.getEmbeddingMatrix(), name="output_embedding")
    # biasWordEmbedding = EmbeddingLayer(words, biasWordEmbedding.getEmbeddingMatrix(), name="bias_embedding")
    # biasFlatten = FlattenLayer(biasWordEmbedding,1)

    scoreUnormalized = DotLayer(act1, outputWordEmbeddingLayer, b=None, useDiagonal=True)

    # linear2 = LinearLayer(act1, hiddenLayerSize, 1,
    #                       weightInitialization=ZeroWeightGenerator(),
    #                       name="linear_softmax_regresion")

    act2 = ActivationLayer(scoreUnormalized, sigmoid)
    # We clip the output of -sigmoid, because this output can be 0  and ln(0) is infinite, which can cause problems.
    output = T.flatten(T.clip(act2.getOutput(), 10 ** -5, 1 - 10 ** -5))

    # Loss Functions
    negativeSamplingLoss = T.nnet.binary_crossentropy(output, yUns).sum()

    # Supervised
    xSup = T.lmatrix("word_window")
    ySup = T.lvector("labels_sup")

    inputSupWordEmbeddingLayer = EmbeddingLayer(xSup, inputWordEmbeddingLayer.getParameters()[0], trainable=True,
                                                name="embedding")
    flattenSup = FlattenLayer(inputSupWordEmbeddingLayer)

    linearSup1 = LinearLayer(flattenSup, wordEmbeddingSize * windowSize, hiddenLayerSize, W=linear1.getParameters()[0],
                             b=linear1.getParameters()[1], name="linear1")
    actSup1 = ActivationLayer(linearSup1, tanh)

    linearSup2 = LinearLayer(actSup1, hiddenLayerSize, labelLexicon.getLen(),
                             weightInitialization=ZeroWeightGenerator(),
                             name="linear_softmax")

    act2Sup = ActivationLayer(linearSup2, softmax)

    predictionSup = ArgmaxPrediction(1).predict(act2Sup.getOutput())
    supervisedLoss = NegativeLogLikelihood().calculateError(act2Sup.getOutput(), predictionSup, ySup)

    # Set training
    inputUnsGenerators = [
        WordWindowGenerator(windowSize, wordLexicon, [], startSymbol, endSymbol)]
    outputUnsGenerators = [ConstantLabel(labelLexicon=None, label=1)]

    trainUnsIterator = SyncBatchIterator(trainUnsupervisedReader, inputUnsGenerators, outputUnsGenerators, batchSize,
                                         shuffle)

    # Lendo Filtros do wnn
    log.info("Lendo filtros básicos")
    wordFilters = getFilters(args.word_filters, log)

    #Supervised Iterator
    inputSupGenerators = [
        WordWindowGenerator(windowSize, wordLexicon, wordFilters, startSymbol, endSymbol)]
    outputSupGenerators = [LabelGenerator(labelLexicon)]

    trainDatasetReader = TokenLabelReader(args.train_supervised, args.token_label_separator)

    trainSupIterator = SyncBatchIterator(trainDatasetReader, inputSupGenerators, outputSupGenerators, batchSize,
                                         shuffle)
    # Metrics
    unsTrainMetrics = [
        LossMetric("lossTrain", negativeSamplingLoss),
    ]

    supTrainMetrics = [
        AccuracyMetric("AccTrain", ySup, predictionSup),
    ]
    #
    # if args.enable_activation_statistics:
    #     activationMetric = ActivationMetric("ActHidden", act1.getOutput(), np.linspace(-1, 1, 21), "avg")
    #     trainMetrics.append(activationMetric)
    #
    # if args.enable_derivative_statistics:
    #     derivativeIntervals = [-float("inf"), -1, -10 ** -1, -10 ** -2, -10 ** -3, -10 ** -4, -10 ** -5, -10 ** -6,
    #                            -10 ** -7,
    #                            -10 ** -8, 0, 10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2,
    #                            10 ** -1, 1,
    #                            float("inf")]
    #     derivativeMetric = DerivativeMetric("DerivativeActHidden", negativeSamplingLoss, act1.getOutput(),
    #                                         derivativeIntervals, "avg")
    #     trainMetrics.append(derivativeMetric)

    allLayers = act2.getLayerSet()
    allLayers.add(linearSup2)
    globalLoss = negativeSamplingLoss + args.lambda_loss * supervisedLoss

    if args.decay.lower() == "word2vec_default":
        isWord2vecDecay = True
        decay = 0.0
    elif args.decay.lower() == "divide_epoch":
        isWord2vecDecay = False
        decay = 1.0
    else:
        decay = -None

    if args.adagrad:
        log.info("Using Adagrad")
        opt = Adagrad(lr=lr, decay=decay)
    else:
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)

    model = NegativeSamplingModelWithSupervised(args.t, noiseRate, sampler, minLr, numExUpdLr, totalNumOfTokens,
                                                numEpochs,
                                                [xUns, words, xSup], [yUns, ySup],
                                                allLayers, opt, globalLoss, unsTrainMetrics, supTrainMetrics,
                                                isWord2vecDecay)
    # Save Model
    if args.save_model:
        savePath = args.save_model
        objsToSave = list(act2.getLayerSet()) + [wordLexicon]

        modelWriter = ModelWriter(savePath, objsToSave, args, parametersToSaveOrLoad)

    cb = []

    # if args.enable_activation_statistics:
    #     cb.append(MetricCB(activationMetric, 500000))
    #
    # if args.enable_derivative_statistics:
    #     cb.append(MetricCB(derivativeMetric, 500000))

    # Training
    model.train([trainUnsIterator, trainSupIterator], numEpochs=numEpochs, callbacks=cb)

    if args.save_model:
        modelWriter.save()


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = dict2obj(JsonArgParser(PARAMETERS).parse(sys.argv[1]))
    mainWnnNegativeSampling(parameters)
