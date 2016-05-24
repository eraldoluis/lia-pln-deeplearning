#!/usr/bin/env python
# -*- coding: utf-8 -*-
import Queue
import random
import threading


class SyncReader(object):
    def __init__(self, generator, shuffle=True):
        generator.setShuffle(False)
        self.__batches = []
        self.__batchIdxs = []
        self.__current = 0
        self.__shuffle = shuffle

        for idx, batch in enumerate(generator):
            self.__batchIdxs.append(idx)
            self.__batches.append(batch)

    def __iter__(self):
        return self

    def next(self):
        if self.__current < len(self.__batchIdxs):
            idx = self.__batchIdxs[self.__current]
            self.__current += 1

            return self.__batches[self.__current]
        else:
            if self.shuffle:
                random.shuffle(self.__batchIdxs)

            self.__current = 0

            raise StopIteration()



class AsyncReader(object):
    def __init__(self, generator, shuffle=True, maxqSize=10, waitTime=0.05):
        generator.setShuffle(shuffle)
        self.__queue, self.__stop = self.generatorQueue(generator, maxqSize, waitTime)

    def next(self):
        b = self.__queue.get()

        if b is None:
            raise StopIteration()

        return b

    def generatorQueue(generator, max_q_size=10, wait_time=0.05):
        '''Builds a threading queue out of a data generator.
        Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
        '''
        q = Queue()
        _stop = threading.Event()

        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        try:
                            generator_output = generator.next()
                        except StopIteration:
                            generator_output = None
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception as e:
                    _stop.set()
                    print e
                    raise

        thread = threading.Thread(target=data_generator_task)
        thread.daemon = True
        thread.start()

        return q, _stop
