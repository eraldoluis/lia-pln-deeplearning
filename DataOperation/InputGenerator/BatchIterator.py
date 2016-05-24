#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Queue
import random
import threading

from datetime import time

import numpy


class BatchAssembler:
    '''
    Create and returns the training batches.
    '''

    def __init__(self, reader, inputGenerators, outputGenerator, batchSize):
        '''
        :type reader: DataOperation.DatasetReader.DatasetReader
        :param reader:

        :type inputGenerators: list[DataOperation.InputGenerator.FeatureGenerator.FeatureGenerator]
        :param inputGenerators: generate the input of the training


        :type outputGenerator: list[DataOperation.InputGenerator.FeatureGenerator.FeatureGenerator]
        :param outputGenerator: generate the output of the training

        :param batchSize: If this parameter has a negative or zero value,
            so this algorithm will consider that the training has variable batch.
            Thus each output of TrainingInputGenerator will consider as the example from a same batch.

            If this parameter has a value bigger than zero, so will packed automatically each element of output in a batch.
        '''

        self.__reader = reader
        self.__inputGenerators = inputGenerators
        self.__outputGenerators = outputGenerator
        self.__batchSize = batchSize


    def getGeneratorObject(self):
        '''
        :return: a generator from the yield expression. This generator will return the batches from the dataset.
        '''
        batch = []
        inputs = [[] for inputGenerator in self.__inputGenerators]
        outputs = []

        for attributes, label in self.__reader.read():
            generatedInputs = []

            for inputGenerator in self.__inputGenerators:
                generatedInputs.append(inputGenerator.generate(attributes))

            generatedOutputs = self.__outputGenerators.generate(label)

            if self.__batchSize > 0:
                for idx in range(len(generatedOutputs)):
                    for idxGen, genInput in enumerate(generatedInputs):
                        inputs[idxGen].append(genInput[idx])

                    outputs.append(generatedOutputs[idx])

                    if len(outputs) == self.__batchSize:
                        yield self.formatToNumpy(inputs, outputs)

                        inputs = [[] for inputGenerator in self.__inputGenerators]
                        outputs = []
            else:
                inputs = generatedInputs
                outputs = generatedOutputs

                yield self.formatToNumpy(inputs, outputs)

        if len(outputs):
            yield self.formatToNumpy(inputs, outputs)

    def formatToNumpy(self, inputs, outputs):
        for idx, input in enumerate(inputs):
            inputs[idx] = numpy.asarray(input)
        return [inputs, numpy.asarray(outputs)]


class SyncBatchIterator(object):
    '''
    Reader all data from data set and generate the training input at once.
    '''

    def __init__(self, reader, inputGenerators, outputGenerator, batchSize, shuffle=True):
        '''
        :type reader: DataOperation.DatasetReader.DatasetReader
        :param reader:

        :type inputGenerators: list[DataOperation.InputGenerator.FeatureGenerator.FeatureGenerator]
        :param inputGenerators: generate the input of the training


        :type outputGenerator: list[DataOperation.InputGenerator.FeatureGenerator.FeatureGenerator]
        :param outputGenerator: generate the output of the training

        :param batchSize: If this parameter has a negative or zero value,
            so this algorithm will consider that the training has variable batch.
            Thus each TrainingInputGenerator has to pass the batch through the generate method.
            If this parameter has a value bigger than zero, so will treat treat each output of generate method as a normal example.

        :param shuffle: is to shufle or not the batches
        '''
        self.__batches = []
        self.__batchIdxs = []
        self.__shuffle = shuffle
        self.__current = 0

        idx = 0
        for batch in BatchAssembler(reader, inputGenerators, outputGenerator, batchSize).getGeneratorObject():
            self.__batches.append(batch)
            self.__batchIdxs.append(idx)
            idx += 1

        if self.__shuffle:
            random.shuffle(self.__batchIdxs)

    def __iter__(self):
        return self

    def next(self):
        if self.__current < len(self.__batchIdxs):
            idx = self.__batchIdxs[self.__current]
            self.__current += 1

            return self.__batches[idx]
        else:
            if self.__shuffle:
                random.shuffle(self.__batchIdxs)

            self.__current = 0

            raise StopIteration()


class AsyncBatchIterator(object):
    def __init__(self, datasetReader, batchSize, inputsGenerator, outputGenerator, shuffle=True, maxqSize=10,
                 waitTime=0.05):
        self.__batchIterator = BatchAssembler(datasetReader, batchSize, inputsGenerator, outputGenerator)
        self.__generatorObj = self.__batchIterator.getGeneratorObject()
        self.__queue, self.__stop = self.generatorQueue(maxQsize, waitTime)

    def __iter__(self):
        return self

    def next(self):
        b = self.__queue.get()

        if b is None:
            raise StopIteration()

        return b

    def generatorQueue(self, maxQsize=10, waitTime=0.05):
        '''Builds a threading queue out of a data generator.
        Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
        '''
        q = Queue()

        _stop = threading.Event()

        def data_generator_task():
            heldData = []

            while not _stop.is_set():
                try:
                    if q.qsize() < maxQsize:
                        try:
                            generator_output = self.__generatorObj.next()

                            if self.__shuffle:
                                c = random.randint(0, 1)
                            else:
                                c = 1

                            if c:
                                q.put(generator_output)
                            else:
                                heldData.append(generator_output)

                                if len(heldData) == min(2, 100 / self.__batchSize):
                                    c = random.randint(0, len(heldData) - 1)
                                    q.put(heldData.pop(c))

                        except StopIteration:
                            random.shuffle(heldData)

                            for data in heldData:
                                q.put(generator_output)

                            q.put(None)

                            self.__generatorObj = self.__batchIterator.getGeneratorObject()
                    else:
                        time.sleep(waitTime)
                except Exception as e:
                    _stop.set()
                    print e
                    raise

        thread = threading.Thread(target=data_generator_task)
        thread.daemon = True
        thread.start()

        return q, _stop
