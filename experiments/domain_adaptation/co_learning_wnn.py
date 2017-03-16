#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import logging
import logging.config
import os
import random
import sys

import numpy as np
import theano
import theano.tensor as T
from pandas import json

from args.JsonArgParser import JsonArgParser
from data.Lexicon import Lexicon
from data.BatchIterator import SyncBatchIterator
from data.Embedding import Embedding
from data.LabelGenerator import LabelGenerator
from data.TokenDatasetReader import TokenLabelReader, TokenReader
from data.WordWindowGenerator import WordWindowGenerator
from model.ModelWriter import ModelWriter
from model.SaveModelCallback import SaveModelCallback
from model.Callback import Callback, DevCallback
from model.CoLearningModel import CoLearningModel
from model.Metric import LossMetric, AccuracyMetric
from model.Objective import NegativeLogLikelihood
from model.Prediction import ArgmaxPrediction
from nnet.ActivationLayer import ActivationLayer, softmax, tanh
from nnet.EmbeddingLayer import EmbeddingLayer
from nnet.FlattenLayer import FlattenLayer
from nnet.LinearLayer import LinearLayer
from nnet.WeightGenerator import ZeroWeightGenerator, GlorotUniform
from optim.Adagrad import Adagrad
from optim.SGD import SGD
from persistence.H5py import H5py
from util.jsontools import dict2obj
from util.util import getFilters

CO_LEARNING_PARAMETERS = {

    # Required
    "token_label_separator": {"required": True,
                              "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "word_filters": {"required": True,
                     "desc": "list contains the filter which is going to be applied in the words. "
                             "Each filter is describe by your module name + . + class name"},
    "lambda_uns": {"desc": "the value of the variable which multiplies the unsupervised loss. This variable controls "
                           "how much the training is affected by the  unsupervised loss.", "required": True},

    # Lexicons
    "label_file": {"desc": "file with all possible labels"},

    # Dataset
    "train_supervised": {"desc": "Supervised Training File Path"},
    "train_unsupervised": {"desc": "Unsupervised Training File Path"},
    "dev": {"desc": "Development File Path"},
    "aux_devs": {
        "desc": "The parameter 'dev' represents the main dev and this parameter represents the auxiliary devs that"
                " will be use to evaluate the model."},
    "test": {"desc": "The Test File Path or a list of file paths"},

    # Save and load
    "save_model": {"desc": "Path + basename that will be used to save the weights and embeddings to be saved."},
    "load_model": {"desc": "Path + basename that will be used to save the weights and embeddings to be loaded."},
    "save_by_acc": {"default": True,
                    "desc": "If this parameter is true, so the script will save the model with the best acc in dev."
                            "However, if its value is false, so the script will save the model at the end of training."},

    # Training parameters
    "loss_uns_epoch": {"default": 0},
    "lr": {"desc": "List with the learning rate values of each classifiers."},
    "l2": {"default": [None, None]},

    "shuffle": {"default": True, "desc": "able or disable the shuffle of training examples."},
    "adagrad": {"desc": "Activate AdaGrad updates.", "default": False},
    "decay": {"default": "DIVIDE_EPOCH",
              "desc": "Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available"},
    "batch_size": {"desc": "a list that contains the batch size of each classifier", "default": [1, 1]},
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},

    # Basic NN parameters
    "normalization": {"desc": "Choose the normalize method to be applied on  word embeddings. "
                              "The possible values are: minmax or mean"},
    "hidden_size": {"default": 300, "desc": "The number of neurons in the hidden layer"},
    "word_window_size": {"default": 5, "desc": "The size of words for the wordsWindow"},
    "with_hidden": {"default": [True,True],
                    "desc": "A list of boolean values. If the value of a index i is true, so the classifier i will have hidden layer, "
                            "otherwise the classifier i won't have a hidden layer."
                    },
    "word_embeddings": {"desc": "A list that contains the word embedding path of each classifier."},
    # "hidden_activation_function": {"default": "tanh",
    #                               "desc": "the activation function of the hidden layer. The possible values are: tanh and sigmoid"},

    # Other parameter
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc": ""},
}


class ChangeLambda(Callback):
    def __init__(self, lambdaShared, lambdaValue, lossUnsupervisedEpoch):
        super(ChangeLambda, self).__init__()

        self.lambdaShared = lambdaShared
        self.lambdaValue = lambdaValue
        self.lossUnsupervisedEpoch = lossUnsupervisedEpoch

    def onEpochBegin(self, epoch, logs={}):
        e = 1 if epoch >= self.lossUnsupervisedEpoch else 0
        self.lambdaShared.set_value(self.lambdaValue * e)


class LossCallback(Callback):
    def __init__(self, loss1, loss2, input1, input2, y):
        super(LossCallback, self).__init__()

        self.f = theano.function([input1, input2, y], [loss1, loss2])
        self.batch = 0
        self.outputs = [0.0, 0.0]

    def onBatchBegin(self, batch, logs={}):
        inputs = []
        inputs += batch[:-2]

        if len(inputs) != 3:
            return

        outputs = self.f(*inputs)
        batchSize = len(batch[1])
        self.batch += batchSize

        for i, output in enumerate(outputs):
            self.outputs[i] += output * batchSize

    def onEpochEnd(self, epoch, logs={}):
        for i in range(len(self.outputs)):
            self.outputs[i] /= self.batch

        print self.outputs

        self.batch = 0
        self.outputs = [0.0, 0.0]


class AccCallBack(Callback):
    def __init__(self, predictionWnn1, predictionWNN2, input1, input2, unsurpervisedDataset):
        super(AccCallBack, self).__init__()

        y = T.lvector()

        agreement = T.mean(T.eq(predictionWnn1, predictionWNN2))
        acc1 = T.mean(T.eq(predictionWnn1, y))
        acc2 = T.mean(T.eq(predictionWNN2, y))

        self.f = theano.function([input1, input2, y], [acc1, acc2, agreement])

        self.batchIterator = unsurpervisedDataset

        # Reading supervised and unsupervised data sets.
        # datasetReader = TokenLabelReader(unsurpervisedDataset, tokenLabelSeparator)
        # batchIterator = SyncBatchIterator(datasetReader, [inputGenerator1, inputGenerator2],
        #                                   outputGenerator, sys.maxint)

    def onEpochEnd(self, epoch, logs={}):

        total = 0
        o = [0.0, 0.0, 0.0]

        for x, y in self.batchIterator:
            inputs = []
            inputs += x
            inputs += y
            batch = len(y[0])
            total += batch
            outputs = self.f(*inputs)

            for i in range(len(outputs)):
                o[i] += outputs[i] * batch

        for i in range(len(o)):
            o[i] = o[i] / total

        print o


def mainColearning(args):
    ################################################
    # Initializing parameters
    ##############################################
    log = logging.getLogger(__name__)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # All parameters which must be saved or loaded in a model
    parametersToSaveOrLoad = {"word_filters", "alg", "word_window_size", "hidden_size", "use_capitalization",
                              "start_symbol",
                              "end_symbol", "with_hidden"}

    # Load model parameters
    if args.load_model:
        persistentManager = H5py(args.load_model)
        savedParameters = json.loads(persistentManager.getAttribute("parameters"))

        if savedParameters.get("charwnn_filters", None) is not None:
            savedParameters["char_filters"] = savedParameters["charwnn_filters"]
            savedParameters.pop("charwnn_filters")
            print savedParameters

        log.info("Loading parameters of the model")
        args = args._replace(**savedParameters)

    # Print arguments
    log.info(str(args))

    # Attribute parameters to variables
    wordWindowSize = args.word_window_size
    hiddenLayerSize = args.hidden_size
    batchSize = args.batch_size
    startSymbol = args.start_symbol
    numEpochs = args.num_epochs
    lrs = args.lr
    normalizeMethod = args.normalization.lower() if args.normalization is not None else None
    withHidden = args.with_hidden

    # Lendo Filtros do wnn
    log.info("Lendo filtros básicos")
    wordFilters = getFilters(args.word_filters, log)

    # Lendo Filtros do charwnn
    # log.info("Lendo filtros do charwnn")
    # charFilters = getFilters(args.char_filters, log)

    # Lendo Filtros do suffix
    # log.info("Lendo filtros do sufixo")
    # suffixFilters = getFilters(args.suffix_filters, log)

    # Lendo Filtros da capitalização
    # log.info("Lendo filtros da capitalização")
    # capFilters = getFilters(args.cap_filters, log)

    ##########################################################################
    # Initialize word embedding and word lexicon
    ##########################################################################

    # Create word lexicon and word embedding
    if args.load_model:
        # Load word lexicon and word embedding of the first classifier
        wordLexiconC1 = Lexicon.fromPersistentManager(persistentManager, "word_lexicon_c1")
        vectors = EmbeddingLayer.getEmbeddingFromPersistenceManager(persistentManager, "word_embedding_layer_c1")

        wordEmbeddingC1 = Embedding(wordLexiconC1, vectors)

        # Load word lexicon and word embedding of the second classifier
        wordLexiconC2 = Lexicon.fromPersistentManager(persistentManager, "word_lexicon_c2")
        vectors = EmbeddingLayer.getEmbeddingFromPersistenceManager(persistentManager, "word_embedding_layer_c2")

        wordEmbeddingC2 = Embedding(wordLexiconC2, vectors)
    elif args.word_embeddings:
        # Load word lexicon and word embedding of the first classifier
        wordLexiconC1, wordEmbeddingC1 = Embedding.fromWord2Vec(args.word_embeddings[0], "UUUNKKK", "word_lexicon_C1")

        # Load word lexicon and word embedding of the second classifier
        wordLexiconC2, wordEmbeddingC2 = Embedding.fromWord2Vec(args.word_embeddings[1], "UUUNKKK", "word_lexicon_C2")
    else:
        log.error("You need to set one of these parameters: load_model or word_embeddings")
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
        wordEmbeddingC1.minMaxNormalization()
        wordEmbeddingC2.minMaxNormalization()
    elif normalizeMethod == "mean":
        log.info("Normalization: mean normalization")
        wordEmbeddingC1.meanNormalization()
        wordEmbeddingC2.meanNormalization()
    else:
        log.error("Unknown normalization method: %s" % normalizeMethod)
        sys.exit(1)

    if normalizeMethod is not None and args.load_model is not None:
        log.warn("The word embedding of model was normalized. This can change the result of test.")

    ##########################################################################
    # Build neural network
    ##########################################################################
    """
    Treinamos dois classificadores, sendo cada um destes uma rede neural própria (WNN)
        que utiliza diferentes word embeddings como entrada.
    Durante o treinamento, é sorteado a cada iteração um exemplo de um dataset artifical,
        formado por exemplos anotados e não anotados.
    Se o exemplo sorteado é anotado então a rede tenta minizar a loss do treinamento supervisionado(ls) dos classificadores
        senão a rede tenta minimizar a loss do treinamento não supervisionado (lu) dos classificadores.
    """

    # Theano variables
    """
    Os classificadores tem entradas separadas, pois
        os indíces das palavras são diferentes nos word embedings.
    """
    inputC1 = T.lmatrix(name="input1")
    inputC2 = T.lmatrix(name="input2")
    y = T.lvector("y")
    _lambdaShared = theano.shared(value=args.lambda_uns, name='lambda', borrow=True)

    # First Classifier
    embeddingLayerC1 = EmbeddingLayer(inputC1, wordEmbeddingC1.getEmbeddingMatrix(), trainable=True,
                                      name="word_embedding_layer_c1")
    flattenC1 = FlattenLayer(embeddingLayerC1)

    if withHidden[0]:
        linearC11 = LinearLayer(flattenC1, wordWindowSize * wordEmbeddingC1.getEmbeddingSize(), hiddenLayerSize,
                                weightInitialization=GlorotUniform(), name="linear_c11")
        actC11 = ActivationLayer(linearC11, tanh)

        layerBeforeSoftmaxC1 = actC11
        sizeLayerBeforeSoftmaxC1 = hiddenLayerSize
    else:
        layerBeforeSoftmaxC1 = flattenC1
        sizeLayerBeforeSoftmaxC1 = wordWindowSize * wordEmbeddingC1.getEmbeddingSize()

    linearC12 = LinearLayer(layerBeforeSoftmaxC1, sizeLayerBeforeSoftmaxC1, labelLexicon.getLen(),
                            weightInitialization=ZeroWeightGenerator(), name="linear_c21")
    actC12 = ActivationLayer(linearC12, softmax)

    # Second Classifier
    embeddingLayerC2 = EmbeddingLayer(inputC2, wordEmbeddingC2.getEmbeddingMatrix(), trainable=True,
                                      name="word_embedding_layer_c1")
    flattenC2 = FlattenLayer(embeddingLayerC2)

    if withHidden[1]:
        linearC21 = LinearLayer(flattenC2, wordWindowSize * wordEmbeddingC2.getEmbeddingSize(), hiddenLayerSize,
                                weightInitialization=GlorotUniform(), name="linear_c21")
        actC21 = ActivationLayer(linearC21, tanh)

        layerBeforeSoftmaxC2 = actC21
        sizeLayerBeforeSoftmaxC2 = hiddenLayerSize
    else:
        layerBeforeSoftmaxC2 = flattenC2
        sizeLayerBeforeSoftmaxC2 = wordWindowSize * wordEmbeddingC2.getEmbeddingSize()

    linearC22 = LinearLayer(layerBeforeSoftmaxC2, sizeLayerBeforeSoftmaxC2, labelLexicon.getLen(),
                            weightInitialization=ZeroWeightGenerator(), name="linear_c22")
    actC22 = ActivationLayer(linearC22, softmax)

    # Output and Prediction
    outputC1 = actC12.getOutput()
    predictionC1 = ArgmaxPrediction(1).predict(outputC1)

    outputC2 = actC22.getOutput()
    predictionC2 = ArgmaxPrediction(1).predict(outputC2)

    # LOSS
    ls1 = NegativeLogLikelihood().calculateError(outputC1, predictionC1, y)
    ls2 = NegativeLogLikelihood().calculateError(outputC2, predictionC2, y)

    # Regularization L2
    if args.l2[0] and withHidden[0]:
        _lambda1 = args.l2[0]
        log.info("Using L2 with lambda= %.2f", _lambda1)
        ls1 += _lambda1 * (T.sum(T.square(linearC11.getParameters()[0])))

    if args.l2[1] and withHidden[1]:
        _lambda2 = args.l2[1]
        log.info("Using L2 with lambda= %.2f", _lambda2)
        ls2 += _lambda2 * (T.sum(T.square(linearC21.getParameters()[0])))

    """
    L = ls +  lambda * lu
    """
    ls = ls1 + ls2
    lu = _lambdaShared * (
        NegativeLogLikelihood().calculateError(outputC1, predictionC1, predictionC2) +
        NegativeLogLikelihood().calculateError(outputC2, predictionC2, predictionC1))

    # Prediction
    """
    Predição combinada dos dois classificadores.
    Fazemos média da saída lineares dos classificadores e, só depois, aplicamos o softmax.
    """
    output = T.stack([linearC12.getOutput(), linearC22.getOutput()])
    average = T.mean(output, 0)
    prediction = ArgmaxPrediction(1).predict(ActivationLayer(average, softmax).getOutput())

    ###########################################################
    # Training and testing
    ###################################################

    # Generators
    inputGenerator1 = WordWindowGenerator(wordWindowSize, wordLexiconC1, wordFilters, startSymbol)
    inputGenerator2 = WordWindowGenerator(wordWindowSize, wordLexiconC2, wordFilters, startSymbol)
    outputGenerator = LabelGenerator(labelLexicon)
    inputGenerators = [inputGenerator1, inputGenerator2]

    # Reading supervised and unsupervised data sets.
    if args.train_supervised and args.train_unsupervised:
        log.info("Reading training examples")

        trainSupervisedDatasetReader = TokenLabelReader(args.train_supervised, args.token_label_separator)
        supervisedBatchIterator = SyncBatchIterator(trainSupervisedDatasetReader,
                                                    inputGenerators,
                                                    [outputGenerator], batchSize[0])

        trainUnsupervisedDataset = TokenReader(args.train_unsupervised)
        unsupervisedBatchIterator = SyncBatchIterator(trainUnsupervisedDataset,
                                                      inputGenerators,
                                                      None, batchSize[1])

        # Get dev inputs and output
        if args.dev:
            log.info("Reading development examples")
            devDatasetReader = TokenLabelReader(args.dev, args.token_label_separator)
            devBatchIterator = SyncBatchIterator(devDatasetReader, inputGenerators, [outputGenerator],
                                                 sys.maxint, shuffle=False)
        else:
            devBatchIterator = None
    else:
        supervisedBatchIterator = None
        unsupervisedBatchIterator = None
        devBatchIterator = None

    # Decay
    if args.decay.lower() == "normal":
        decay = 0.0
    elif args.decay.lower() == "divide_epoch":
        decay = 1.0

    # Optimization algorithms. Each classifier has your own learning rate
    if args.adagrad:
        log.info("Using Adagrad")
        optmizers = [Adagrad(lr=lrs[0], decay=decay),Adagrad(lr=lrs[1], decay=decay)]
    else:
        log.info("Using SGD")
        optmizers = [SGD(lr=lrs[0], decay=decay), SGD(lr=lrs[1], decay=decay)]

    # Metrics
    supervisedTrainMetrics = [
        LossMetric("SupervisedLoss", ls, True),
        LossMetric("SupervisedLossC1", ls1, True),
        LossMetric("SupervisedLossC2", ls2, True),
        AccuracyMetric("AccTrain", y, prediction)
    ]

    unsupervisedTrainMetrics = [
        LossMetric("UnsupervisedLoss", lu, True),
        AccuracyMetric("Agreement", predictionC1, predictionC2),
    ]

    evalMetrics = [
        LossMetric("LossDev", ls, True),
        AccuracyMetric("AccDev", y, prediction),
    ]

    testMetrics = [
        LossMetric("LossTest", ls, True),
        AccuracyMetric("AccTest", y, prediction),
    ]

    # Create the model
    classifierLayers = [actC22.getLayerSet(), actC12.getLayerSet()]

    colearninModel = CoLearningModel([inputC1, inputC2], y, classifierLayers, optmizers, prediction, ls, lu, args.loss_uns_epoch,
                                     supervisedTrainMetrics, unsupervisedTrainMetrics, evalMetrics, testMetrics, )

    # Training
    if supervisedBatchIterator and unsupervisedBatchIterator:
        callback = []

        if args.save_model:
            savePath = args.save_model
            objsToSave = list(actC22.getLayerSet()) + list(actC12.getLayerSet()) + [wordLexiconC1, wordLexiconC2,
                                                                                    labelLexicon]

            modelWriter = ModelWriter(savePath, objsToSave, args, parametersToSaveOrLoad)

            # Save the model with best acc in dev
            if args.save_by_acc:
                callback.append(SaveModelCallback(modelWriter, evalMetrics[1], "accuracy", True))

        if args.aux_devs:
            callback.append(
                DevCallback(colearninModel, args.aux_devs, args.token_label_separator, inputGenerators,
                            [outputGenerator]))

        log.info("Training")
        colearninModel.train([supervisedBatchIterator, unsupervisedBatchIterator], numEpochs, devBatchIterator,
                             callbacks=callback)

        # Save the model at the end of training
        if args.save_model and not args.save_by_acc:
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
            testReader = SyncBatchIterator(testDatasetReader, inputGenerators, [outputGenerator], sys.maxint,
                                           shuffle=False)

            log.info("Testing")
            colearninModel.test(testReader)

            if args.print_prediction:
                f = codecs.open(args.print_prediction, "w", encoding="utf-8")

                for x, labels in testReader:
                    inputs = x

                    predictions = colearninModel.prediction(inputs)

                    for prediction in predictions:
                        f.write(labelLexicon.getLexicon(prediction))
                        f.write("\n")


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = dict2obj(JsonArgParser(CO_LEARNING_PARAMETERS).parse(sys.argv[1]))
    mainColearning(parameters)
