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
from data.Lexicon import Lexicon
from data.TokenDatasetReader import TokenReader
from data.WordWindowGenerator import WordWindowGenerator
from model.Callback import Callback
from model.Metric import LossMetric
from model.NegativeSamplingModel import NegativeSamplingModel
from model.Prediction import ArgmaxPrediction
from nnet.ActivationLayer import ActivationLayer, tanh, sigmoid
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.FlattenLayer import FlattenLayer
from nnet.LinearLayer import LinearLayer
from nnet.WeightGenerator import ZeroWeightGenerator
from optim.SGD import SGD
from util.Sampler import Sampler
from util.jsontools import dict2obj

PARAMETERS = {
    # Required
    "train": {"required": True, "desc": "Use text data from file to train the model",},
    "lr": {"required": True, "desc": "Set the starting learning rate"},
    "hidden_size": {"required": True, "desc": "size of the hidden layer"},
    "num_epochs": {"required": True,
                   "desc": "Number of epochs: how many iterations over the training set."},
    "noise_rate": {"required": True, "desc": "Number of negative examples",},

    # Word embedding options
    "word_embedding_size": {
        "desc": "the size of the word embedding. This parameter will be used when word_embedding is None."},

    # Model options
    "window_size": {"default": 5, "desc": "size of the window size"},

    # Other options
    "power": {"default": 0.75, "desc": "q(w)^power, where q(w) is the unigram distribution."},
    "min_count": {"default": 5,
                  "desc": "This will discard words that appear less than n times",},
    "decay": {"default": "DIVIDE_EPOCH",
              "desc": "Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available"},

    "t": {"default": 10 ** -5,
          "desc": "Set threshold for occurrence of words. Those that appear with higher frequency in the training data "
                  "will be randomly down-sampled; default is 1e-5, useful range is (0, 1e-10)"},

    "shuffle": {"default": True, "desc": "able or disable the shuffle of training examples."},

    "batch_size": {"default": 1},

    "start_symbol": {"default": "</s>",
                     "desc": "Object that will be place when the initial limit of list is exceeded"},
    # "end_symbol": {"default": "</s>",
    #                "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc": ""},

}


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
    batchSize = args.batch_size
    shuffle = args.shuffle
    lr = args.lr
    numEpochs = args.num_epochs
    power = args.power

    log = logging.getLogger(__name__)

    log.info(args)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.decay.lower() == "normal":
        decay = 0.0
    elif args.decay.lower() == "divide_epoch":
        decay = 1.0

    # Calculate the frequency of each word
    trainReader = TokenReader(args.train)
    wordLexicon = Lexicon("UUKNNN")
    wordLexicon.put(startSymbol, False)

    for tokens, labels in trainReader.read():
        # Word2vec considers that the number of lines is the frequency of </s>
        tokens += [startSymbol]

        for token in tokens:
            wordLexicon.put(token)

    # Prune the words with the frequency less than min_count
    wordLexicon.prune(args.min_count)

    # Calculte the unigram distribution
    frequency = np.power(wordLexicon.getFrequencyOfAllWords(), power)
    total = float(frequency.sum())

    # # Print the distribution of all words
    # for _ in xrange(len(frequency)):
    #     print "%s\t%d\t%.4f" % (wordLexicon.getLexicon(_), frequency[_],frequency[_]/float(total))

    sampler = Sampler(frequency / float(total))


    # Create a random embedding for each word
    wordEmbedding = Embedding(wordLexicon, None, wordEmbeddingSize)
    log.info("Lexicon size: %d" % (wordLexicon.getLen()))

    # Create NN
    x = T.lmatrix("word_window")
    y = T.lvector("labels")

    wordEmbeddingLayer = EmbeddingLayer(x, wordEmbedding.getEmbeddingMatrix(), name="embedding")
    flatten = FlattenLayer(wordEmbeddingLayer)

    linear1 = LinearLayer(flatten, wordEmbeddingSize * windowSize, hiddenLayerSize, name="linear1")
    act1 = ActivationLayer(linear1, tanh)

    # Softmax regression. It's like a logistic regression
    linear2 = LinearLayer(act1, hiddenLayerSize, 1,
                          weightInitialization=ZeroWeightGenerator(),
                          name="linear_softmax_regresion")

    act2 = ActivationLayer(linear2, sigmoid)
    output = T.flatten(act2.getOutput())

    # Loss Function
    # We clip the output of -log, because this output can be 0  and ln(0) is infinite, which can cause problems.
    negativeSamplingLoss = (y * T.clip(-T.log(output), 0, 1000) + (1. - y) * T.clip(-T.log(1 - output), 0, 1000)).sum()

    # Set training
    inputGenerators = [
        WordWindowGenerator(windowSize, wordLexicon, [], startSymbol, endSymbol)]

    outputGenerators = [ConstantLabel(labelLexicon=None, label=1)]

    trainIterator = SyncBatchIterator(trainReader, inputGenerators, outputGenerators, batchSize, shuffle)

    trainMetrics = [
        LossMetric("lossTrain", negativeSamplingLoss)
    ]

    allLayers = act2.getLayerSet()

    opt = SGD(lr=lr, decay=decay)

    model = NegativeSamplingModel(args.t, noiseRate, sampler, [x], [y], allLayers, opt, negativeSamplingLoss,
                                  trainMetrics)

    # Training
    model.train(trainIterator, numEpochs=numEpochs, callbacks=[])


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = dict2obj(JsonArgParser(PARAMETERS).parse(sys.argv[1]))
    mainWnnNegativeSampling(parameters)
