#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from data.BatchIterator import SyncBatchIterator
from data.TokenDatasetReader import TokenLabelReader


class Callback(object):
    """Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def __init__(self):
        pass

    def onEpochBegin(self, epoch, logs={}):
        pass

    def onEpochEnd(self, epoch, logs={}):
        pass

    def onBatchBegin(self, batch, logs={}):
        pass

    def onBatchEnd(self, batch, logs={}):
        pass

    def onTrainBegin(self, logs={}):
        pass

    def onTrainEnd(self, logs={}):
        pass


class DevCallback(Callback):
    """
    Apply the model in auxiliar development datasets.
    """

    def __init__(self, model, devs, tokenLabelSep, inputGenerators, outputGenerators):
        """

        :param model: Model class
        :param devs: list with the file path of each development datasets
        :param tokenLabelSep: specify the character that is being used to separate the token from the label in the dataset.
        :param inputGenerators: a list of InputGenerators
        :param outputGenerators: a list of OutputGenerators
        """
        self.__datasetIterators = []
        self.__model = model

        for devFile in devs:
            devDatasetReader = TokenLabelReader(devFile, tokenLabelSep)
            devIterator = SyncBatchIterator(devDatasetReader, inputGenerators, outputGenerators, sys.maxint,
                                            shuffle=False)

            self.__datasetIterators.append(devIterator)

    def onEpochEnd(self, epoch, logs={}):
        for it in self.__datasetIterators:
            self.__model.test(it)


