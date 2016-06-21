import importlib
import logging
import os

import sys

import numpy
from theano import tensor as T

from DataOperation.Embedding import EmbeddingFactory, RandomUnknownStrategy
from DataOperation.InputGenerator.BatchIterator import SyncBatchIterator, AsyncBatchIterator
from DataOperation.InputGenerator.WindowGenerator import WindowGenerator
from DataOperation.TokenDatasetReader import TokenReader
from ModelOperation.Model import Model
from ModelOperation.Objective import MeanSquaredError
from ModelOperation.SaveModelCallback import ModelWriter, SaveModelCallback
from NNet import EmbeddingLayer, FlattenLayer, DropoutLayer, LinearLayer, ActivationLayer
from NNet.ActivationLayer import sigmoid, ActivationLayer, tanh
from NNet.DropoutLayer import DropoutLayer
from NNet.EmbeddingLayer import EmbeddingLayer
from NNet.FlattenLayer import FlattenLayer
from NNet.LinearLayer import LinearLayer
from NNet.TiedLayer import TiedLayer
from NNet.WeightGenerator import SigmoidGlorot, GlorotUniform
from Optimizers import SGD
from Optimizers.SGD import SGD
from Parameters.JsonArgParser import JsonArgParser
import logging.config

DA_PARAMETERS = {
    "train": {"desc": "Training File Path", "required": True},
    "num_epochs": {"required": True, "desc": "Number of epochs: how many iterations over the training set."},
    "lr": {"desc": "learning rate value", "required": True},
    "filters": {"required": True,
                "desc": "list contains the filters. Each filter is describe by your module name + . + class name"},
    "word_embedding": {"desc": "word embedding File Path", "required": True},

    "sync": {"desc": "choose if the whole train will be loaded to the memory or not", "default": True},
    "noise_rate": {"desc": "noise rate", "default": 0.3},
    "encoder_size": {"desc": "the size of hidden layer that receive the input", "default": 300},
    "word_window_size": {"default": 5, "desc": "The size of words for the wordsWindow"},
    "load_model": {"desc": "Model File Path to be loaded."},
    "save_model": {"desc": "Model File Path to be saved."},
    "batch_size": {"default": 16},
    "word_emb_size": {"default": 100, "desc": "size of word embedding"},
    "start_symbol": {"default": "</s>", "desc": "Object that will be place when the initial limit of list is exceeded"},
    "end_symbol": {"default": "</s>", "desc": "Object that will be place when the end limit of list is exceeded"},
}


class DAModelWritter(ModelWriter):
    def __init__(self, savePath, encodeLayer, decodeLayer):
        '''
        :param savePath: path where the model will be saved

        :type encodeLayer: NNet.LinearLayer.LinearLayer
        :type decodeLayer: NNet.TiedLayer.TiedLayer
        '''
        self.__savePath = savePath
        self.__encodeLayer = encodeLayer
        self.__decodeLayer = decodeLayer
        self.__logging = logging.getLogger(__name__)

    def save(self):
        weights = {}

        W1, b1 = self.__encodeLayer.getParameters()
        weights["W_Encoder"] = W1.get_value()
        weights["b_Encoder"] = b1.get_value()

        b2 = self.__decodeLayer.getParameters()[0]

        weights["b_Decoder"] = b2.get_value()

        numpy.save(self.__savePath, weights)

        self.__logging.info("Model Saved")


def main(**kwargs):
    log = logging.getLogger(__name__)
    log.info(kwargs)

    lr = kwargs["lr"]
    wordWindowSize = kwargs["word_window_size"]
    startSymbol = kwargs["start_symbol"]
    endSymbol = kwargs["end_symbol"]
    numEpochs = kwargs["num_epochs"]
    encoderSize = kwargs["encoder_size"]
    batchSize = kwargs["batch_size"]
    noiseRate = kwargs["noise_rate"]
    saveModel = kwargs["save_model"]
    sync = kwargs["sync"]
    seed = None

    filters = []

    for filterName in kwargs["filters"]:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    log.info("Reading W2v File")
    embedding = EmbeddingFactory().createFromW2V(kwargs["word_embedding"], RandomUnknownStrategy())
    embedding.meanNormalization()

    datasetReader = TokenReader(kwargs["train"])
    inputGenerator = WindowGenerator(wordWindowSize, embedding, filters,
                                     startSymbol, endSymbol)

    if sync:
        log.info("Loading e pre-processing train data set")
        trainBatchGenerator = SyncBatchIterator(datasetReader, [inputGenerator], None, batchSize)
    else:
        trainBatchGenerator = AsyncBatchIterator(datasetReader, [inputGenerator], None, batchSize)
        # We can't stop, because the data set is reading in a asynchronous way
        # embedding.stopAdd()

    input = T.lmatrix("window_words")

    # Window of words
    embeddingLayer = EmbeddingLayer(input, embedding.getEmbeddingMatrix(), trainable=False)
    flatten = FlattenLayer(embeddingLayer)

    # Noise  Layer
    dropoutOutput = DropoutLayer(flatten, noiseRate, seed)

    # Encoder
    linear1 = LinearLayer(dropoutOutput, wordWindowSize * embedding.getEmbeddingSize(), encoderSize,
                          weightInitialization=GlorotUniform())
    act1 = ActivationLayer(linear1, tanh)

    # Decoder
    linear2 = TiedLayer(act1, linear1.getParameters()[0], wordWindowSize * embedding.getEmbeddingSize())
    act2 = ActivationLayer(linear2, tanh)

    # Input of the hidden layer
    x = flatten.getOutput()

    # Creates the model
    mdaModel = Model(input, x, True)

    sgd = SGD(lr, decay=0.0)
    prediction = act2
    loss = MeanSquaredError().calculateError(act2, prediction, x)

    log.info("Compiling the model")
    mdaModel.compile(act2.getLayerSet(),sgd, prediction, loss)

    cbs = []

    if saveModel:
        writter = DAModelWritter(saveModel, linear1, linear2)
        cbs.append(SaveModelCallback(writter, "loss", False))
    log.info("Traning model")
    mdaModel.train(trainBatchGenerator, numEpochs, callbacks=cbs)


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = JsonArgParser(DA_PARAMETERS).parse(sys.argv[1])
    main(**parameters)
