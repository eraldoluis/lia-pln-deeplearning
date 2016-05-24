#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import logging
import sys

import keras
from keras.layers.containers import Sequential
from keras.layers.embeddings import Embedding

from DataOperation import Lexicon
from DataOperation.Embedding import EmbeddingFactory
from DataOperation.FormatDataSupervisedTrain import WNNInputBuilder
from NNet import EmbeddingLayer
from Parameters import JsonArgParser
from util.util import loadConfigLogging


WNN_PARAMETERS = u'''
{
    "train": {"desc": "Training File Path", required: true},
    "num_epochs": {"required": "true", "desc": "Number of epochs: how many iterations over the training set." },
    "token_label_separator": { "required": true, "desc": "specify the character that is being used to separate the token from the label in the dataset."},
    "lr": {"desc":"learning rate value", "required": true},
    "filters": {"required":true, "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},

    "test": {"desc": "Test File Path"},
    "dev": {"desc": "Development File Path"},
    "alg": {"default":"window_word", "desc": "The type of algorithm to train and test"},
    "hidden_size": {"default": 300, "desc": "The number of neurons in the hidden layer"},
    "word_window_size": {"default": 5 , "desc": "The size of words for the wordsWindow" },
    "batch_size": {"default": 16},
    "word_emb_size": {"default": 100, "desc": "size of word embedding"},
    "word_embedding": {"desc": "word embedding File Path"},
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
    "seed": {"desc":""},
    "adagrad": {"desc": "Activate AdaGrad updates.", default: true},
    "decay": {default: "normal", "desc": "Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available"}

}
'''

def main(**kwargs):
    log = logging.getLogger(__name__)
    log.info(kwargs)

    if kwargs["alg"] == "window_stn":
        isSentenceModel = True
    elif kwargs["alg"] == "window_word":
        isSentenceModel = False
    else:
        raise Exception("The value of model_type isn't valid.")

    filters = []

    for filterName in kwargs["filters"]:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    if kwargs["word_embedding"]:
        log.info("Reading W2v File")
        embedding = EmbeddingFactory().createFromW2V(kwargs["word_embedding"])
    else:
        embedding = EmbeddingFactory().createEmptyEmbedding(kwargs["word_emb_size"])

    # Get the inputs and output
    labelLexicon = Lexicon()
    prepareDataToAlg = WNNInputBuilder(kwargs["word_window_size"], kwargs["start_symbol"], kwargs["end_symbol"])

    log.info("Reading training examples")

    trainExamples = prepareDataToAlg.readTokenLabelFile(kwargs["train"], embedding, labelLexicon, filters,
                                                        kwargs["token_label_separator"], isSentenceModel)
    embedding.stopAdd()
    labelLexicon.stopAdd()

    log.info("Using %d examples from train data set" % (len(trainExamples[0])))

    # Get dev inputs and output
    dev = kwargs["dev"]

    if dev:
        log.info("Reading development examples")
        devExamples = prepareDataToAlg.readTokenLabelFile(dev, embedding, labelLexicon, filters,
                                                          kwargs["label_tkn_sep"], isSentenceModel)
        log.info("Using %d examples from development data set" % (len(devExamples[0])))
    else:x
        devExamples = None

    if isSentenceModel:
        raise NotImplementedError("Model of sentence window was't implemented yet.")
    else:











                wnn = Sequential()
        Embedding

        wnn.add(EmbeddingLayer(embedding.getNumberOfEmbeddings(), embedding.getEmbeddingSize(),
                          weights=[numpy.asarray(embedding.getEmbeddingMatrix())],
                          input_length=windowSize))
        wnn.add(Flatten())

        wnn.add(Dense(hiddenLayerSize))
        wnn.add(Activation("tanh"))

        wnn.add(Dense(labelLexicon.getLen()))
        wnn.add(Activation("softmax"))
        opt = SGD(lr=lr)
        wnn.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbs = [SaveModelByDevAcc(pathModel, wnn)] if pathModel else []
    wnn.fit(trainExamples[0], trainExamples[1], nb_epoch=numEpochs, batch_size=batchSize, callbacks=cbs, verbose=2,
            validation_data=devExamples)





if __name__ == '__main__':
    loadConfigLogging()

    parameters = JsonArgParser(WNN_PARAMETERS).parse(sys.argv[1])

    main(parameters)







