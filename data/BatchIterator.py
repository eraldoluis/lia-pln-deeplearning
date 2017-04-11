#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Queue
import logging
import numpy
import random
import threading
import sys


class BatchAssembler:
    """
    Create and return training (mini) batches.
    """

    def __init__(self, reader, inputGenerators, outputGenerators, batchSize):
        """
        :type reader: data.DatasetReader.DatasetReader
        :param reader:

        :type inputGenerators: list[data.InputGenerator.FeatureGenerator.FeatureGenerator]
        :param inputGenerators: generate the input of the training

        :type outputGenerators: list[DataOperation.InputGenerator.FeatureGenerator.FeatureGenerator]
        :param outputGenerators: generate the output of the training

        :param batchSize: If this parameter has a negative or zero value,
            so this algorithm will consider that the training has variable batch.
            Thus each output of TrainingInputGenerator will consider a same batch.
            If this parameter has a value bigger than zero, so will packed automatically each element of output in a batch.
        """

        self.__reader = reader
        self.__inputGenerators = inputGenerators
        self.__outputGenerators = outputGenerators
        self.__batchSize = batchSize
        self.__printed = False
        self.__log = logging.getLogger(__name__)

    def getGeneratorObject(self):
        """
        :return: yield-based generator that generates training batches from the data set.
        """
        inputs = [[] for _ in xrange(len(self.__inputGenerators))]
        if self.__outputGenerators:
            outputs = [[] for _ in xrange(len(self.__outputGenerators))]
        else:
            outputs = []

        numExamples = 0
        for attributes, label in self.__reader.read():
            generatedInputs = []
            generatedOutputs = []

            for inputGenerator in self.__inputGenerators:
                generatedInputs.append(inputGenerator(attributes))

            # Unsupervised networks do not use an output (like autoencoders, for instance).
            if self.__outputGenerators:
                for outputGenerator in self.__outputGenerators:
                    generatedOutputs.append(outputGenerator(label))

            if self.__batchSize > 0:
                numExamples += len(generatedInputs[0])
                # Batch  has fixed size
                for idx in xrange(len(generatedInputs[0])):
                    for idxGen, genInput in enumerate(generatedInputs):
                        inputs[idxGen].append(genInput[idx])

                    # Unsupervised networks do not use an output (like autoencoders, for instance).
                    if self.__outputGenerators:
                        for idxGen, genOutput in enumerate(generatedOutputs):
                            outputs[idxGen].append(genOutput[idx])

                    if len(inputs[0]) == self.__batchSize:
                        yield self.__formatToNumpy(inputs, outputs)

                        inputs = [[] for _ in xrange(len(self.__inputGenerators))]
                        if self.__outputGenerators:
                            outputs = [[] for _ in xrange(len(self.__outputGenerators))]
            else:
                numExamples += 1
                # Batch doesn't have fixed size
                yield self.__formatToNumpy(generatedInputs, generatedOutputs)

        # The remaining batches are returned
        if (len(inputs) > 0 and len(inputs[0]) > 0) or (len(outputs) > 0 and len(outputs[0]) > 0):
            yield self.__formatToNumpy(inputs, outputs)

        if not self.__printed:
            self.__log.info("Number of examples: %d" % numExamples)

    @staticmethod
    def __formatToNumpy(inputs, outputs):
        for idx, inp in enumerate(inputs):
            inputs[idx] = numpy.asarray(inp)
        for idx, out in enumerate(outputs):
            outputs[idx] = numpy.asarray(out)
        return inputs, outputs


class SyncBatchIterator(object):
    """
    Reads all data from data set and generates the training input at once.
    """

    def __init__(self, reader, inputGenerators, outputGenerator, batchSize, shuffle=True):
        """
        :type reader: data.DatasetReader.DatasetReader
        :param reader:

        :type inputGenerators: list[data.InputGenerator.FeatureGenerator.FeatureGenerator]
        :param inputGenerators: generate the input of the training

        :type outputGenerator: list[data.InputGenerator.FeatureGenerator.FeatureGenerator]
        :param outputGenerator: generate the output of the training

        :param batchSize: If this parameter has a negative or zero value,
            so this algorithm will consider that the training has variable batch.
            Thus each TrainingInputGenerator has to pass the batch through the generate method.
            If this parameter has a value bigger than zero, so will treat treat each output of generate method as a normal example.

        :param shuffle: shufle or not the batches
        """
        self.__batches = []
        self.__batchIdxs = []
        self.__shuffle = shuffle
        self.__current = 0
        self.__log = logging.getLogger(__name__)
        self.__batchSize = batchSize

        idx = 0
        for batch in BatchAssembler(reader, inputGenerators, outputGenerator, batchSize).getGeneratorObject():
            self.__batches.append(batch)
            self.__batchIdxs.append(idx)
            idx += 1

        self.__size = len(self.__batchIdxs)
        self.__log.info("Number of batches: %d" % self.__size)
        self.__log.info("BatchSize: %d" % batchSize)

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

    def get(self, idx):
        return self.__batches[idx]

    def getBatchSize(self, idx):
        return self.__batchSize

    def size(self):
        return self.__size


class AsyncBatchIterator(object):
    """
    Reads a certain quantity of batches from the data set at a time.
    """

    def __init__(self, reader, inputGenerators, outputGenerator, batchSize, shuffle=False, maxqSize=1000):
        """
            :type reader: data.DatasetReader.DatasetReader
            :param reader:

            :type inputGenerators: list[data.InputGenerator.FeatureGenerator.FeatureGenerator]
            :param inputGenerators: generate the input of the training

            :type outputGenerator: list[data.InputGenerator.FeatureGenerator.FeatureGenerator]
            :param outputGenerator: generate the output of the training

            :param batchSize: If this parameter has a negative or zero value,
                so this algorithm will consider that the training has variable batch.
                Thus each TrainingInputGenerator has to pass the batch through the generate method.
                If this parameter has a value bigger than zero, so will treat treat each output of generate method as a normal example.

            :param shuffle: is to shufle or not the batches

            :param maxqSize: maximum number of batches in queue
            """
        self.__batchAssembler = BatchAssembler(reader, inputGenerators, outputGenerator, batchSize)
        self.__generatorObj = self.__batchAssembler.getGeneratorObject()
        self.__queue, self.__stop = self.__generatorQueue(maxqSize)
        self.__shuffle = shuffle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__stop.set()
        # This is a trick to unblock the producer thread in case it is blocked because the queue is full.
        self.__queue.get_nowait()

    def __iter__(self):
        return self


    def next(self):
        try:
            b = self.__queue.get()
            self.__queue.task_done()
        except Queue.Empty as e:
            sys.stderr.write("Empty!\n%s\n" % e)

        if not b:
            raise StopIteration()

        return b

    def __generatorQueue(self, maxQsize):
        # Queue of batches
        q = Queue.Queue(maxQsize)

        _stop = threading.Event()

        def data_generator_task():
            # Batches outside of the queue
            heldData = []

            while not _stop.is_set():
                try:
                    batch = self.__generatorObj.next()

                    if not self.__shuffle:
                        q.put(batch)
                    else:
                        # Flip a coin to decide whether the batch will be put in the queue or in the held data.
                        if random.random() < 1.0 / maxQsize:
                            q.put(batch)
                        else:
                            heldData.append(batch)
                            # If heldData has some specific length, so one element of this array will be put on queue.
                            if len(heldData) == maxQsize:
                                c = random.randint(0, len(heldData) - 1)
                                q.put(heldData.pop(c))

                except StopIteration as e:
                    if self.__shuffle:
                        random.shuffle(heldData)
                    for data in heldData:
                        q.put(data)

                    # Signal the end of batches.
                    q.put(None)

                    self.__generatorObj = self.__batchAssembler.getGeneratorObject()

        # Create thread that will read the batches
        thread = threading.Thread(target=data_generator_task)
        thread.daemon = True
        thread.start()

        return q, _stop
