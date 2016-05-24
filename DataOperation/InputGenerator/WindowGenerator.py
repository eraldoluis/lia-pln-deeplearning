from DataOperation import Context
from DataOperation.InputGenerator.FeatureGenerator import FeatureGenerator


class WindowGenerator(FeatureGenerator):
    '''
    Generate window of words from each word of a list.
    This list can be a line.
    '''

    def __init__(self, windowSize, embedding, filters,
                 startPadding, endPadding=None):
        '''
        :type windowSize: int
        :param windowSize: the size of window

        :type embedding: DataOperation.Embedding.Embedding
        :param embedding:

        :type filters: list[DataOperation.Filters.Filter]
        :param filters:

        :param startPadding: Object that will be place when the initial limit of a list is exceeded

        :param endPadding: Object that will be place when the end limit a list is exceeded.
            If this parameter is None, so the endPadding has the same value of startPadding
        '''
        self.__window = Context.Window(windowSize)
        self.__embedding = embedding
        self.__filters = filters

        self.__startPaddingIdx, self.__endPaddingIdx = self.checkPadding(startPadding, endPadding, embedding)

    def generate(self, rawData):
        '''
        Receives a list of tokens and returns window of words.
        If isSentenceModel is true, than all window of words will be returned as one example.
        Else each window of words will be returned as one example

        :type rawData: list[basestring]
        :param rawData: a list of tokens
        :return:
        '''
        tokens = rawData
        tknIdxs = []
        y = []

        for token in tokens:
            for f in self.__filters:
                token = f.filter(token)

            tknIdxs.append(self.__embedding.put(token))

        x = []
        windowGen = self.__window.buildWindows(tknIdxs, self.__startPaddingIdx, self.__endPaddingIdx)

        for window in windowGen:
            x.append(window)

        return x

    def checkPadding(self, startPadding, endPadding, embedding):
        '''
        Verify if the start padding and end padding exist in lexicon or embedding.

        :param startPadding: Object that will be place when the initial limit of a list is exceeded

        :param endPadding: Object that will be place when the end limit a list is exceeded.
            If this parameter is None, so the endPadding has the same value of startPadding

        :param embedding: DataOperation.Embedding.Embedding

        :return: the index of start and end padding in lexicon
        '''

        if not embedding.exist(startPadding):
            if embedding.isStopped():
                raise Exception("Start Padding doens't exist")

            startPaddingIdx = embedding.put(startPadding)
        else:
            startPaddingIdx = embedding.getLexiconIndex(startPadding)
        if endPadding is not None:
            if not embedding.exist(endPadding):
                if embedding.isStopped():
                    raise Exception("End Padding doens't exist")

                endPaddingIdx = embedding.put(endPadding)
            else:
                endPaddingIdx = embedding.getLexiconIndex(endPadding)
        else:
            endPaddingIdx = None

        return startPaddingIdx, endPaddingIdx