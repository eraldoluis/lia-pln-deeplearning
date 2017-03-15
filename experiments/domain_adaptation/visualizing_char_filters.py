#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import json
import logging
import logging.config
import operator
import os
import sys
from itertools import izip

import numpy
import theano
import theano.tensor as T

from args.JsonArgParser import JsonArgParser
from data.BatchIterator import SyncBatchIterator
from data.CapitalizationFeatureGenerator import CapitalizationFeatureGenerator
from data.CharacterWindowGenerator import CharacterWindowGenerator
from data.Embedding import Embedding
from data.LabelGenerator import LabelGenerator
from data.Lexicon import Lexicon
from data.SuffixFeatureGenerator import SuffixFeatureGenerator
from data.TokenDatasetReader import TokenLabelReader
from data.WordWindowGenerator import WordWindowGenerator
from model.Prediction import ArgmaxPrediction
from nnet.ActivationLayer import ActivationLayer, softmax, tanh, sigmoid
from nnet.ConcatenateLayer import ConcatenateLayer
from nnet.EmbeddingConvolutionalLayer import EmbeddingConvolutionalLayer
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.FlattenLayer import FlattenLayer
from nnet.LinearLayer import LinearLayer
from nnet.WeightGenerator import ZeroWeightGenerator, GlorotUniform, SigmoidGlorot
from persistence.H5py import H5py
from util.jsontools import dict2obj
from util.util import getFilters

WNN_PARAMETERS = {
    # Required
    "token_label_separator": {"required": True,
                              "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "charwnn_model": {"required": True, "desc": "The file of charwnn"},
    "charwnn_prediction": {"required": True, "desc": ""},
    "test": {"required": True, "desc": "The file with right answers"},
    "print_char_window_activate_filters": {"default": True, "desc": "The file with right answers"},

    # STUB
    "word_filters": {},
    "suffix_filters": {},
    "char_filters": {},
    "cap_filters": {},
    "alg": {},
    "hidden_activation_function": {},
    "word_window_size": {},
    "with_charwnn": {},
    "conv_size": {},
    "charwnn_with_act": {},
    "suffix_size": {},
    "use_capitalization": {},
    "start_symbol": {},
    "end_symbol": {}
}


class ListWithBestCharacterWindow(object):
    def __init__(self, maxSize=40):
        self.maxSize = maxSize
        self.lowestValue = sys.maxint
        self.indexLowestValue = -1
        self.list = []
        self.windows = {}

    def add(self, characterWindow, value, append):
        k = characterWindow

        if self.windows.get(k, False):
            return

        if len(self.list) < self.maxSize:
            if value < self.lowestValue:
                self.lowestValue = value
                self.indexLowestValue = len(self.list)

            self.list.append((value, k, append))
            self.windows[k] = True
        else:
            if value > self.lowestValue:
                self.windows.pop(self.list[self.indexLowestValue][1])

                self.list[self.indexLowestValue] = (value, k, append)
                self.windows[k] = True

                self.lowestValue = sys.maxint
                self.indexLowestValue = -1

                for i, tu in enumerate(self.list):
                    if tu[0] < self.lowestValue:
                        self.lowestValue = tu[0]
                        self.indexLowestValue = i

    def getAllSorted(self):
        return sorted(self.list, key=lambda item: item[0], reverse=True)


def mainWnn(args):
    ################################################
    # Initializing parameters
    ##############################################
    log = logging.getLogger(__name__)

    parametersToSaveOrLoad = {"word_filters", "suffix_filters", "char_filters", "cap_filters",
                              "alg", "hidden_activation_function", "word_window_size",
                              "with_charwnn", "conv_size", "charwnn_with_act", "suffix_size", "use_capitalization",
                              "start_symbol", "end_symbol"}

    # Load parameters of the saving model
    persistentManager = H5py(args.charwnn_model)
    savedParameters = json.loads(persistentManager.getAttribute("parameters"))

    log.info("Loading parameters of the model")
    args = args._replace(**savedParameters)

    log.info(args)

    # Read the parameters
    startSymbol = args.start_symbol
    endSymbol = args.end_symbol
    wordWindowSize = args.word_window_size
    hiddenActFunctionName = args.hidden_activation_function

    withCharWNN = args.with_charwnn
    charWindowSize = args.char_window_size
    hiddenLayerSize = args.hidden_size

    useSuffixFeatures = args.suffix_size > 0
    useCapFeatures = args.use_capitalization

    # Insert the character that will be used to fill the matrix
    # with a dimension lesser than chosen dimension.This enables that the convolution is performed by a matrix multiplication.
    startSymbolChar = "</s>"
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
    wordLexicon = Lexicon.fromPersistentManager(persistentManager, "word_lexicon")
    vectors = EmbeddingLayer.getEmbeddingFromPersistenceManager(persistentManager, "word_embedding_layer")

    wordEmbedding = Embedding(wordLexicon, vectors)
    embeddingSize = wordEmbedding.getEmbeddingSize()

    # Read char lexicon and create char embeddings
    if withCharWNN:
        charLexicon = Lexicon.fromPersistentManager(persistentManager, "char_lexicon")
        vectors = EmbeddingConvolutionalLayer.getEmbeddingFromPersistenceManager(persistentManager,
                                                                                 "char_convolution_layer")

        charEmbedding = Embedding(charLexicon, vectors)
        charEmbeddingSize = charEmbedding.getEmbeddingSize()
    else:
        # Read suffix lexicon if suffix size is greater than 0
        if useSuffixFeatures:
            suffixLexicon = Lexicon.fromPersistentManager(persistentManager, "suffix_lexicon")
            vectors = EmbeddingConvolutionalLayer.getEmbeddingFromPersistenceManager(persistentManager,
                                                                                     "suffix_embedding")
            suffixEmbedding = Embedding(suffixLexicon, vectors)

        # Read capitalization lexicon
        if useCapFeatures:
            capLexicon = Lexicon.fromPersistentManager(persistentManager, "cap_lexicon")
            vectors = EmbeddingConvolutionalLayer.getEmbeddingFromPersistenceManager(persistentManager,
                                                                                     "cap_embedding")

            capEmbedding = Embedding(capLexicon, vectors)

    # Read labels
    labelLexicon = Lexicon.fromPersistentManager(persistentManager, "label_lexicon")

    wordWindow = T.lmatrix("word_window")
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
    hiddenActFunction = method_name(hiddenActFunctionName)

    weightInit = SigmoidGlorot() if hiddenActFunction == sigmoid else GlorotUniform()

    linear1 = LinearLayer(layerBeforeLinear, sizeLayerBeforeLinear, hiddenLayerSize,
                          weightInitialization=weightInit, name="linear1")
    act1 = ActivationLayer(linear1, hiddenActFunction)

    linear2 = LinearLayer(act1, hiddenLayerSize, labelLexicon.getLen(), weightInitialization=ZeroWeightGenerator(),
                          name="linear_softmax")
    act2 = ActivationLayer(linear2, softmax)
    prediction = ArgmaxPrediction(1).predict(act2.getOutput())

    # Load the model
    alreadyLoaded = set([wordEmbeddingLayer])

    for o in (act2.getLayerSet() - alreadyLoaded):
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

    outputGenerator = LabelGenerator(labelLexicon)

    # Printing embedding information
    dictionarySize = wordEmbedding.getNumberOfVectors()

    log.info("Size of  word dictionary and word embedding size: %d and %d" % (dictionarySize, embeddingSize))

    if withCharWNN:
        log.info("Size of  char dictionary and char embedding size: %d and %d" % (
            charEmbedding.getNumberOfVectors(), charEmbedding.getEmbeddingSize()))

    log.info("Reading test examples")

    rawLexicon = Lexicon("UNNNKKK")
    rawWindowGenerator = WordWindowGenerator(wordWindowSize, rawLexicon, [], startSymbol, endSymbol)
    inputGenerators.append(rawWindowGenerator)

    testDatasetReader = TokenLabelReader(args.test, args.token_label_separator)
    testReader = SyncBatchIterator(testDatasetReader, inputGenerators, [outputGenerator], 1, shuffle=False)

    rawLexicon.stopAdd()

    maxSize = 150
    filterResults = [ListWithBestCharacterWindow(maxSize) for _ in xrange(convSize)]

    if args.print_char_window_activate_filters:
        filterFunction = theano.function(inputs=[charWindowIdxs], outputs=charEmbeddingConvLayer.o.getOutput())

        # Print the N characters window with the greatest value for each filter.
        for x, labels in testReader:
            for label, window, charwnnInput, rawWindow in izip(labels[0], x[0], x[1], x[2]):
                outConvFilters = filterFunction([charwnnInput])
                rawLabel = labelLexicon.getLexicon(label)

                allWindow = ""

                for rawIdx in rawWindow:
                    allWindow += rawLexicon.getLexicon(rawIdx) + " "

                halfOfWindow = wordWindowSize / 2
                wordToBePredicted = rawLexicon.getLexicon(rawWindow[halfOfWindow])
                outConvFilter = outConvFilters[halfOfWindow]

                for charWindowIdx, charWindow in enumerate(charwnnInput[halfOfWindow]):
                    characters = ""
                    for charIdx in charWindow:
                        characters += charLexicon.getLexicon(charIdx)

                    for filterIdx in xrange(convSize):
                        filterResults[filterIdx].add(characters, outConvFilter[charWindowIdx][filterIdx],
                                                     (wordToBePredicted, rawLabel, allWindow))

        for idx, filterResult in enumerate(filterResults):
            print "Filtro " + str(idx)

            for a, r in enumerate(filterResult.getAllSorted()):
                value = r[0]
                character = r[1]
                rawWord, label, allWindow = r[2]

                print "%s\t%.5f\t%s\t%s\t%s" % (character, value, rawWord, label, allWindow)
            print ""

    parameterToCalculateGrad = charEmbeddingConvLayer._EmbeddingConvolutionalLayer__embedLayer.getOutput()
    labelIdx = T.lscalar()
    probabilityLabel = linear2.getOutput()[0][labelIdx]

    filterGradientsFunc = theano.function(inputs=[wordWindow, charWindowIdxs, labelIdx],
                                          outputs=[T.grad(probabilityLabel, parameterToCalculateGrad)])

    labelResults = {}

    for i in xrange(labelLexicon.getLen()):
        labelResults[i] = [[] for _ in xrange(convSize)]

    for x, labels in testReader:
        labelIdx = labels[0][0]
        rawLabel = labelLexicon.getLexicon(labelIdx)

        _in = x[:2] + [labelIdx]

        filterGradients = filterGradientsFunc(*_in)
        record = labelResults[labelIdx]

        charwnnInput = x[1][0].reshape((args.word_window_size * numMaxChar, args.char_window_size))
        gradReshaped = filterGradients[0].reshape(args.word_window_size * numMaxChar,
                                                  args.char_window_size * charEmbedding.getEmbeddingSize())
        allWindow = ""
        rawWindow = x[2][0]
        for rawIdx in rawWindow:
            allWindow += rawLexicon.getLexicon(rawIdx) + " "

        print allWindow + "\t" + labelLexicon.getLexicon(labelIdx)

        previous = ""

        for charWindow, gradientWindow in izip(charwnnInput, gradReshaped):
            characters = ""

            for charIdx in charWindow:
                characters += charLexicon.getLexicon(charIdx)

            mean = numpy.absolute(gradientWindow).mean()

            n = characters + " " + str(mean)

            if previous == n and charIdx == charLexicon.getLexiconIndex(artificialChar):
                continue

            if charIdx == charLexicon.getLexiconIndex(artificialChar):
                characters = "ARTIF"

            print characters + "\t",
            print mean

            previous = n

        print ""




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

    parameters = dict2obj(JsonArgParser(WNN_PARAMETERS).parse(sys.argv[1]))
    mainWnn(parameters)
