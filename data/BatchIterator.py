#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Queue
import logging
import numpy
import random
import threading
import time


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
        generatedOutputs = None
        nmExamples = 0

        if self.__outputGenerators:
            outputs = [[] for _ in xrange(len(self.__outputGenerators))]
        else:
            outputs = []

        for attributes, label in self.__reader.read():
            generatedInputs = []
            generatedOutputs = []

            for inputGenerator in self.__inputGenerators:
                generatedInputs.append(inputGenerator(attributes))

            # Unsupervised networks do not use an output (like autoencoders, for instance).
            if self.__outputGenerators:
                for outputGenerator in self.__outputGenerators:
                    generatedOutputs.append(outputGenerator.generate(label))

            nmExamples += len(generatedInputs[0])

            if self.__batchSize > 0:
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
                # Batch don't have fixed size
                inputs = generatedInputs
                outputs = generatedOutputs

                yield self.__formatToNumpy(inputs, outputs)

        # The remaining batches are returned
        if len(inputs[0]):
            yield self.__formatToNumpy(inputs, outputs)

        if not self.__printed:
            self.__log.info("Number of examples: %d" % nmExamples)

    def __formatToNumpy(self, inputs, outputs):
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

        :param shuffle: is to shufle or not the batches
        """
        self.__batches = []
        self.__batchIdxs = []
        self.__shuffle = shuffle
        self.__current = 0
        self.__log = logging.getLogger(__name__)

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

    def size(self):
        return self.__size


class AsyncBatchIterator(object):
    """
    Reads a certain quantity of batches from the data set at a time.
    """

    def __init__(self, datasetReader, inputGenerators, outputGenerator, batchSize, shuffle=True, maxqSize=100,
                 waitTime=0.005):
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

            :param waitTime: time which de thread is going to wait when the queue is full
            """
        self.__batchIterator = BatchAssembler(datasetReader, inputGenerators, outputGenerator, batchSize)
        self.__generatorObj = self.__batchIterator.getGeneratorObject()
        self.__queue, self.__stop = self.generatorQueue(maxqSize, waitTime)
        self.__shuffle = shuffle
        self.__batchSize = batchSize

    def __iter__(self):
        return self

    def next(self):
        while True:
            try:
                b = self.__queue.get(timeout=0.0001)
                break;
            except Queue.Empty as e:
                print "Empty! %s" % e
                continue

        if b is None:
            raise StopIteration()

        return b

    def generatorQueue(self, maxQsize, waitTime):
        # Queue of batches
        q = Queue.Queue()

        _stop = threading.Event()

        def data_generator_task():
            # Batches outside of the queue
            heldData = []

            while not _stop.is_set():
                try:
                    if q.qsize() < maxQsize:
                        try:
                            generator_output = self.__generatorObj.next()

                            if self.__shuffle:
                                # Flip a coin to decide if the batch will be put in queue or not.
                                c = (random.randint(0, 1) == 1)
                            else:
                                c = True

                            if c:
                                # If coin is true, so this batch will be put in the queue
                                q.put(generator_output)
                            else:
                                heldData.append(generator_output)

                                # If heldData has some specific length, so one element of this array will be put on queue.
                                if len(heldData) == maxQsize:
                                    c = random.randint(0, len(heldData) - 1)
                                    q.put(heldData.pop(c))

                        except StopIteration:
                            random.shuffle(heldData)

                            for data in heldData:
                                q.put(data)

                            q.put(None)

                            self.__generatorObj = self.__batchIterator.getGeneratorObject()
                    else:
                        time.sleep(waitTime)
                except Exception as e:
                    _stop.set()
                    print e
                    raise

        # Create thread that will read the batches
        thread = threading.Thread(target=data_generator_task)
        thread.daemon = True
        thread.start()

        return q, _stop
