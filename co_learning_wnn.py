#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import logging
import logging.config
import os
import random
import sys

import numpy as np
import theano
import theano.tensor as T
from theano import printing

from DataOperation.Embedding import EmbeddingFactory, RandomUnknownStrategy
from DataOperation.InputGenerator.BatchIterator import SyncBatchIterator
from DataOperation.InputGenerator.LabelGenerator import LabelGenerator
from DataOperation.InputGenerator.WindowGenerator import WindowGenerator
from DataOperation.Lexicon import createLexiconUsingFile
from DataOperation.TokenDatasetReader import TokenLabelReader, TokenReader
from ModelOperation.Callback import Callback
from ModelOperation.CoLearningModel import CoLearningModel
from ModelOperation.Model import Model, ModelUnit
from ModelOperation.Objective import NegativeLogLikelihood
from ModelOperation.Prediction import ArgmaxPrediction, CoLearningWnnPrediction
from NNet.ActivationLayer import ActivationLayer, softmax, tanh
from NNet.EmbeddingLayer import EmbeddingLayer
from NNet.FlattenLayer import FlattenLayer
from NNet.LinearLayer import LinearLayer
from NNet.WeightGenerator import ZeroWeightGenerator, GlorotUniform
from Optimizers.Adagrad import Adagrad
from Optimizers.SGD import SGD
from Parameters.JsonArgParser import JsonArgParser

CO_LEARNING_PARAMETERS = {
    "token_label_separator": {"required": True,
                              "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "filters": {"required": True,
                "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},
    "lambda": {"desc": "", "required": True},
    "label_file": {"desc": "", "required": True},
    "batch_size": {"required": True},

    "train_supervised": {"desc": "Supervised Training File Path"},
    "train_unsupervised": {"desc": "Unsupervised Training File Path"},
    "num_epochs": {"desc": "Number of epochs: how many iterations over the training set."},
    "lr": {"desc": "learning rate value"},
    "save_model": {"desc": "Path + basename that will be used to save the weights and embeddings to be saved."},

    "test": {"desc": "Test File Path"},
    "dev": {"desc": "Development File Path"},
    "load_model": {"desc": "Path + basename that will be used to save the weights and embeddings to be loaded."},

    "alg": {"default": "window_word",
            "desc": "The type of algorithm to train and test. The posible inputs are: window_word or window_stn"},
    "hidden_size": {"default": 300, "desc": "The number of neurons in the hidden layer"},
    "word_window_size": {"default": 5, "desc": "The size of words for the wordsWindow"},
    "word_emb_size": {"default": 100, "desc": "size of word embedding"},
    "word_embedding1": {"desc": "word embedding File Path"},
    "word_embedding2": {"desc": "word embedding File Path"},
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc": ""},
    "adagrad": {"desc": "Activate AdaGrad updates.", "default": True},
    "decay": {"default": "DIVIDE_EPOCH",
              "desc": "Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available"},
    "load_hidden_layer": {"desc": "the file which contains weights and bias of pre-trainned hidden layer"},
    "hidden_activation_function": {"default": "tanh",
                                   "desc": "the activation function of the hidden layer. The possible values are: tanh and sigmoid"},
    "shuffle": {"default": True, "desc": "able or disable the shuffle of training examples."},
    "normalization": {"desc": "Choose the normalize method to be applied on  word embeddings. "
                              "The possible values are: max_min, mean_normalization or none"},
    "seed": {"desc": ""},
    "l2": {"default": [None, None]},
    "loss_uns_epoch": {"default": 0}
}


class ChangeLambda(Callback):
    def __init__(self, lambdaShared, lambdaValue, lossUnsupervisedEpoch):
        self.lambdaShared = lambdaShared
        self.lambdaValue = lambdaValue
        self.lossUnsupervisedEpoch = lossUnsupervisedEpoch

    def onEpochBegin(self, epoch, logs={}):
        e = 1 if epoch >= self.lossUnsupervisedEpoch else 0
        self.lambdaShared.set_value(self.lambdaValue * e)


class LossCallback(Callback):
    def __init__(self, loss1, loss2, input1, input2, y):
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


def main(**kwargs):
    log = logging.getLogger(__name__)
    log.info(kwargs)

    if kwargs["seed"] != None:
        random.seed(kwargs["seed"])
        np.random.seed(kwargs["seed"])

    filters = []

    for filterName in kwargs["filters"]:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    # Get the inputs and output
    wordWindowSize = kwargs["word_window_size"]
    hiddenLayerSize = kwargs["hidden_size"]
    batchSize = kwargs["batch_size"]
    startSymbol = kwargs["start_symbol"]
    numEpochs = kwargs["num_epochs"]
    lr = kwargs["lr"]
    labelLexicon = createLexiconUsingFile(kwargs["label_file"])

    log.info("Reading training examples")

    log.info("Reading W2v File1")
    embedding1 = EmbeddingFactory().createFromW2V(kwargs["word_embedding1"], RandomUnknownStrategy())

    # Supervised part
    # Learner1
    input1 = T.lmatrix(name="input1")

    embeddingLayer1 = EmbeddingLayer(input1, embedding1.getEmbeddingMatrix(), trainable=True)
    flatten1 = FlattenLayer(embeddingLayer1)

    linear11 = LinearLayer(flatten1, wordWindowSize * embedding1.getEmbeddingSize(), hiddenLayerSize,
                           weightInitialization=GlorotUniform())
    act11 = ActivationLayer(linear11, tanh)

    linear12 = LinearLayer(act11, hiddenLayerSize, labelLexicon.getLen(), weightInitialization=ZeroWeightGenerator())
    act12 = ActivationLayer(linear12, softmax)

    ## Learner2
    log.info("Reading W2v File2")
    embedding2 = EmbeddingFactory().createFromW2V(kwargs["word_embedding2"], RandomUnknownStrategy())

    input2 = T.lmatrix(name="input2")

    embeddingLayer2 = EmbeddingLayer(input2, embedding2.getEmbeddingMatrix(), trainable=True)
    flatten2 = FlattenLayer(embeddingLayer2)

    linear21 = LinearLayer(flatten2, wordWindowSize * embedding2.getEmbeddingSize(), hiddenLayerSize,
                           weightInitialization=GlorotUniform())
    act21 = ActivationLayer(linear21, tanh)

    linear22 = LinearLayer(act21, hiddenLayerSize, labelLexicon.getLen(), weightInitialization=ZeroWeightGenerator())
    act22 = ActivationLayer(linear22, softmax)

    y = T.lvector("y")

    # Set loss and prediction and retrieve all layers
    output1 = act12.getOutput()
    prediction1 = ArgmaxPrediction(1).predict(output1)
    loss1 = NegativeLogLikelihood().calculateError(output1, prediction1, y)

    if kwargs["l2"][0]:
        _lambda1 = kwargs["l2"][0]
        log.info("Using L2 with lambda= %.2f", _lambda1)
        loss1 += _lambda1 * (T.sum(T.square(linear11.getParameters()[0])))

    output2 = act22.getOutput()
    prediction2 = ArgmaxPrediction(1).predict(output2)
    loss2 = NegativeLogLikelihood().calculateError(output2, prediction2, y)

    if kwargs["l2"][1]:
        _lambda2 = kwargs["l2"][1]
        log.info("Using L2 with lambda= %.2f", _lambda2)
        loss2 += _lambda2 * (T.sum(T.square(linear21.getParameters()[0])))

    loss = loss1 + loss2

    ## CoLearningPrediction
    output = T.stack([linear12.getOutput(), linear22.getOutput()])
    # return T.argmax(output, 2)[T.argmax(T.max(output, 2), 0),T.arange(output.shape[1])]
    average = T.mean(output, 0)
    prediction = ArgmaxPrediction(1).predict(ActivationLayer(average, softmax).getOutput())
    # prediction = CoLearningWnnPrediction().predict([output1, output2])

    supervisedModeUnit = ModelUnit("supervised_wnn", [input1, input2], y, loss, prediction=prediction)

    # Unsupervised part

    ## Learner1
    inputUnsuper1 = T.lmatrix(name="input_unsupervised_1")

    embeddingLayerUnsuper1 = EmbeddingLayer(inputUnsuper1, embeddingLayer1.getParameters()[0], trainable=True)

    flattenUnsuper1 = FlattenLayer(embeddingLayerUnsuper1)

    w, b = linear11.getParameters()
    linearUnsuper11 = LinearLayer(flattenUnsuper1, wordWindowSize * embedding1.getEmbeddingSize(), hiddenLayerSize,
                                  W=w, b=b)
    actUnsupervised11 = ActivationLayer(linearUnsuper11, tanh)

    w, b = linear12.getParameters()
    linearUnsuper12 = LinearLayer(actUnsupervised11, hiddenLayerSize, labelLexicon.getLen(), W=w, b=b)
    actUnsuper12 = ActivationLayer(linearUnsuper12, softmax)

    ## Learner2
    inputUnsuper2 = T.lmatrix(name="input_unsupervised_2")

    embeddingLayerUnsuper2 = EmbeddingLayer(inputUnsuper2, embeddingLayer2.getParameters()[0], trainable=True)
    flattenUnsuper2 = FlattenLayer(embeddingLayerUnsuper2)

    w, b = linear21.getParameters()
    linearUnsuper21 = LinearLayer(flattenUnsuper2, wordWindowSize * embedding2.getEmbeddingSize(), hiddenLayerSize, W=w,
                                  b=b)
    actUnsuper21 = ActivationLayer(linearUnsuper21, tanh)

    w, b = linear22.getParameters()
    linearUnsuper22 = LinearLayer(actUnsuper21, hiddenLayerSize, labelLexicon.getLen(), W=w, b=b)
    actUnsuper22 = ActivationLayer(linearUnsuper22, softmax)

    # Set loss and prediction and retrieve all layers
    outputUns1 = actUnsuper12.getOutput()
    predictionUns1 = ArgmaxPrediction(1).predict(outputUns1)

    outputUns2 = actUnsuper22.getOutput()
    predictionUns2 = ArgmaxPrediction(1).predict(outputUns2)
    #
    # unsupervisedLoss = kwargs["lambda"] * (
    #         NegativeLogLikelihood().calculateError(outputUns1, predictionUns1, predictionUns2) +
    #         NegativeLogLikelihood().calculateError(outputUns2, predictionUns2, predictionUns1))

    _lambdaShared = theano.shared(value=kwargs["lambda"], name='lambda', borrow=True)

    unsupervisedLoss = _lambdaShared * (
        NegativeLogLikelihood().calculateError(outputUns1, predictionUns1, predictionUns2) +
        NegativeLogLikelihood().calculateError(outputUns2, predictionUns2, predictionUns1))

    unsupervisedUnit = ModelUnit("unsupervised_wnn", [inputUnsuper1, inputUnsuper2], None, unsupervisedLoss,
                                 yWillBeReceived=False)

    # Creates model
    model = CoLearningModel()

    model.addTrainingModelUnit(supervisedModeUnit, metrics=["loss", "acc"])
    model.addTrainingModelUnit(unsupervisedUnit, metrics=["loss"])

    model.setEvaluatedModelUnit(supervisedModeUnit, metrics=["acc"])

    # Compile Model
    opt1 = SGD(lr=lr[0], decay=1.0)
    opt2 = SGD(lr=lr[1], decay=1.0)

    log.info("Compiling the model")
    model.compile([(opt1, {supervisedModeUnit: act12.getLayerSet(), unsupervisedUnit: actUnsuper12.getLayerSet()}),
                   (opt2, {supervisedModeUnit: act22.getLayerSet(), unsupervisedUnit: actUnsuper22.getLayerSet()})])

    # Generators
    inputGenerator1 = WindowGenerator(wordWindowSize, embedding1, filters, startSymbol)
    inputGenerator2 = WindowGenerator(wordWindowSize, embedding2, filters, startSymbol)
    outputGenerator = LabelGenerator(labelLexicon)

    # Reading supervised and unsupervised data sets.
    trainSupervisedDatasetReader = TokenLabelReader(kwargs["train_supervised"], kwargs["token_label_separator"])
    trainSupervisedDatasetReader = SyncBatchIterator(trainSupervisedDatasetReader, [inputGenerator1, inputGenerator2],
                                                     [outputGenerator], batchSize[0])

    trainUnsupervisedDataset = TokenLabelReader(kwargs["train_unsupervised"], kwargs["token_label_separator"])
    trainUnsupervisedDatasetReader = SyncBatchIterator(trainUnsupervisedDataset,
                                                       [inputGenerator1, inputGenerator2],
                                                       [outputGenerator], batchSize[1])

    embedding1.stopAdd()
    embedding2.stopAdd()
    labelLexicon.stopAdd()

    # Get dev inputs and output
    log.info("Reading development examples")
    devDatasetReader = TokenLabelReader(kwargs["dev"], kwargs["token_label_separator"])
    devReader = SyncBatchIterator(devDatasetReader, [inputGenerator1, inputGenerator2], [outputGenerator], sys.maxint,
                                  shuffle=False)

    lambdaChange = ChangeLambda(_lambdaShared, kwargs["lambda"], kwargs["loss_uns_epoch"])
    lossCallback = LossCallback(loss1, loss2, input1, input2, y)

    trainUnsupervisedDatasetReaderAcc = SyncBatchIterator(trainUnsupervisedDataset,
                                                          [inputGenerator1, inputGenerator2],
                                                          [outputGenerator], sys.maxint)

    accCallBack = AccCallBack(prediction1, prediction2, input1, input2,
                              unsurpervisedDataset=trainUnsupervisedDatasetReaderAcc)
    # Training Model
    model.train([trainSupervisedDatasetReader, trainUnsupervisedDatasetReader], numEpochs, devReader,
                callbacks=[lambdaChange, accCallBack, lossCallback])


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = JsonArgParser(CO_LEARNING_PARAMETERS).parse(sys.argv[1])
    main(**parameters)
