#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import importlib
import json
import logging
import logging.config
import os
import random
import sys
import time
from time import time

import numpy as np
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
from model.BasicModel import BasicModel
from model.Metric import LossMetric, AccuracyMetric
from model.ModelWriter import ModelWriter
from model.Objective import NegativeLogLikelihood
from model.Prediction import ArgmaxPrediction
from model.SaveModelCallback import SaveModelCallback
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

WNN_PARAMETERS = {
    # Required
    "word_filters": {"required": False,
                     "desc": "a list which contains the filters. Each filter is describe by your module name + . + class name"},

    # Lexicons.
    "label_file": {"desc": "file with all possible labels"},
    "word_lexicon": {"desc": "word lexicon"},

    # Dataset
    "train": {"desc": "Training File Path"},
    "dev": {"desc": "Development File Path"},
    "test": {"desc": "Test File Path"},

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
    "hidden_size": {"default": 300, "desc": "The number of neurons in the hidden layer"},
    "word_emb_size": {"default": 100, "desc": "size of word embedding"},
    "word_embedding": {"desc": "word embedding File Path"},
    "word_window_size": {"default": 5, "desc": "The size of words for the wordsWindow"},

    # Other parameter
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc": "Seed used in the random number generator"},
}

def mainWnn(args):
    ################################################
    # Initializing parameters
    ##############################################
    log = logging.getLogger(__name__)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    log.info(str(args))

    # Read the parameters
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

    log.info("Loading word filters...")
    wordFilters = getFilters(args.word_filters, log)
    
    # Create the lexicon and go out after this
    if args.create_only_lexicon:
        inputGenerators = []
        lexiconsToSave = []

        if args.word_lexicon and not os.path.exists(args.word_lexicon):
            wordLexicon = Lexicon("UUUNKKK", "labelLexicon")

            inputGenerators.append(
                WordWindowGenerator(wordWindowSize, wordLexicon, wordFilters, startSymbol, endSymbol))
            lexiconsToSave.append((wordLexicon, args.word_lexicon))

        if not os.path.exists(args.label_file):
            labelLexicon = Lexicon(None, "labelLexicon")
            outputGenerator = [LabelGenerator(labelLexicon)]
            lexiconsToSave.append((labelLexicon, args.label_file))
        else:
            outputGenerator = None

        if args.char_lexicon and not os.path.exists(args.char_lexicon):
            charLexicon = Lexicon("UUUNKKK", "charLexicon")

            charLexicon.put(startSymbolChar)
            charLexicon.put(artificialChar)

            inputGenerators.append(
                CharacterWindowGenerator(charLexicon, numMaxChar, charWindowSize, wordWindowSize, artificialChar,
                                         startSymbolChar, startPaddingWrd=startSymbol, endPaddingWrd=endSymbol,
                                         filters=charFilters))

            lexiconsToSave.append((charLexicon, args.char_lexicon))

        if args.suffix_lexicon and not os.path.exists(args.suffix_lexicon):
            suffixLexicon = Lexicon("UUUNKKK", "suffixLexicon")

            if args.suffix_size <= 0:
                raise Exception(
                    "Unable to generate the suffix lexicon because the suffix is less than or equal to 0.")

            inputGenerators.append(
                SuffixFeatureGenerator(args.suffix_size, wordWindowSize, suffixLexicon, suffixFilters))

            lexiconsToSave.append((suffixLexicon, args.suffix_lexicon))

        if args.cap_lexicon and not os.path.exists(args.cap_lexicon):
            capLexicon = Lexicon("UUUNKKK", "capitalizationLexicon")

            inputGenerators.append(CapitalizationFeatureGenerator(wordWindowSize, capLexicon, capFilters))

            lexiconsToSave.append((capLexicon, args.cap_lexicon))

        if len(inputGenerators) == 0:
            inputGenerators = None

        if not (inputGenerators or outputGenerator):
            log.info("All lexicons have been generated.")
            return

        trainDatasetReader = TokenLabelReader(args.train, args.token_label_separator)
        trainReader = SyncBatchIterator(trainDatasetReader, inputGenerators, outputGenerator, batchSize,
                                        shuffle=shuffle)

        for lexicon, pathToSave in lexiconsToSave:
            lexicon.save(pathToSave)

        log.info("Lexicons were generated with success!")

        return

    ################################################
    # Starting training
    ###########################################

    if withCharWNN and (useSuffixFeatures or useCapFeatures):
        raise Exception("It's impossible to use hand-crafted features with Charwnn.")

    # Read word lexicon and create word embeddings
    if args.load_model:
        wordLexicon = Lexicon.fromPersistentManager(persistentManager, "word_lexicon")
        vectors = EmbeddingLayer.getEmbeddingFromPersistenceManager(persistentManager, "word_embedding_layer")

        wordEmbedding = Embedding(wordLexicon, vectors)

    elif args.word_embedding:
        wordLexicon, wordEmbedding = Embedding.fromWord2Vec(args.word_embedding, "UUUNKKK", "word_lexicon")
    elif args.word_lexicon:
        wordLexicon = Lexicon.fromTextFile(args.word_lexicon, True, "word_lexicon")
        wordEmbedding = Embedding(wordLexicon, vectors=None, embeddingSize=embeddingSize)
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

    # Build neural network
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

    linear2 = LinearLayer(layerBeforeSoftmax, sizeLayerBeforeSoftmax, labelLexicon.getLen(),
                          weightInitialization=ZeroWeightGenerator(),
                          name="linear_softmax")
    act2 = ActivationLayer(linear2, softmax)
    prediction = ArgmaxPrediction(1).predict(act2.getOutput())

    # Load the model
    if args.load_model:
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

    if args.train:
        log.info("Reading training examples")

        trainDatasetReader = TokenLabelReader(args.train, args.token_label_separator)
        trainReader = SyncBatchIterator(trainDatasetReader, inputGenerators, [outputGenerator], batchSize,
                                        shuffle=shuffle)

        # Get dev inputs and output
        dev = args.dev

        if dev:
            log.info("Reading development examples")
            devDatasetReader = TokenLabelReader(args.dev, args.token_label_separator)
            devReader = SyncBatchIterator(devDatasetReader, inputGenerators, [outputGenerator], sys.maxint,
                                          shuffle=False)
        else:
            devReader = None
    else:
        trainReader = None
        devReader = None

    y = T.lvector("y")

    if args.decay.lower() == "normal":
        decay = 0.0
    elif args.decay.lower() == "divide_epoch":
        decay = 1.0

    if args.adagrad:
        log.info("Using Adagrad")
        opt = Adagrad(lr=lr, decay=decay)
    else:
        log.info("Using SGD")
        opt = SGD(lr=lr, decay=decay)

    # Printing embedding information
    dictionarySize = wordEmbedding.getNumberOfVectors()

    log.info("Size of  word dictionary and word embedding size: %d and %d" % (dictionarySize, embeddingSize))

    if withCharWNN:
        log.info("Size of  char dictionary and char embedding size: %d and %d" % (
            charEmbedding.getNumberOfVectors(), charEmbedding.getEmbeddingSize()))

    if useSuffixFeatures:
        log.info("Size of  suffix dictionary and suffix embedding size: %d and %d" % (
            suffixEmbedding.getNumberOfVectors(), suffixEmbedding.getEmbeddingSize()))

    if useCapFeatures:
        log.info("Size of  capitalization dictionary and capitalization embedding size: %d and %d" % (
            capEmbedding.getNumberOfVectors(), capEmbedding.getEmbeddingSize()))

    # Compiling
    loss = NegativeLogLikelihood().calculateError(act2.getOutput(), prediction, y)

    if args.lambda_L2:
        _lambda = args.lambda_L2
        log.info("Using L2 with lambda= %.2f", _lambda)
        loss += _lambda * (T.sum(T.square(linear1.getParameters()[0])))

    trainMetrics = [
        LossMetric("LossTrain", loss, True),
        AccuracyMetric("AccTrain", y, prediction),
    ]

    evalMetrics = [
        LossMetric("LossDev", loss, True),
        AccuracyMetric("AccDev", y, prediction),
    ]

    testMetrics = [
        LossMetric("LossTest", loss, True),
        AccuracyMetric("AccTest", y, prediction),
    ]

    wnnModel = BasicModel(inputModel, [y], act2.getLayerSet(), opt, prediction, loss, trainMetrics=trainMetrics,
                          evalMetrics=evalMetrics, testMetrics=testMetrics, mode=None)
    # Training
    if trainReader:
        callback = []

        if args.save_model:
            savePath = args.save_model
            objsToSave = list(act2.getLayerSet()) + [wordLexicon, labelLexicon]

            if withCharWNN:
                objsToSave.append(charLexicon)

            if useSuffixFeatures:
                objsToSave.append(suffixLexicon)

            if useCapFeatures:
                objsToSave.append(capLexicon)

            modelWriter = ModelWriter(savePath, objsToSave, args, parametersToSaveOrLoad)

            # Save the model with best acc in dev
            if args.save_by_acc:
                callback.append(SaveModelCallback(modelWriter, evalMetrics[1], "accuracy", True))

        log.info("Training")
        wnnModel.train(trainReader, numEpochs, devReader, callbacks=callback)

        # Save the model at the end of training
        if args.save_model and not args.save_by_acc:
            modelWriter.save()

    # Testing
    if args.test:
        log.info("Reading test examples")
        testDatasetReader = TokenLabelReader(args.test, args.token_label_separator)
        testReader = SyncBatchIterator(testDatasetReader, inputGenerators, [outputGenerator], sys.maxint, shuffle=False)

        log.info("Testing")
        wnnModel.test(testReader)

        if args.print_prediction:
            f = codecs.open(args.print_prediction, "w", encoding="utf-8")

            for x, labels in testReader:
                inputs = x

                predictions = wnnModel.prediction(inputs)

                for prediction in predictions:
                    f.write(labelLexicon.getLexicon(prediction))
                    f.write("\n")


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
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = dict2obj(JsonArgParser(WNN_PARAMETERS).parse(sys.argv[1]))
    mainWnn(parameters)
