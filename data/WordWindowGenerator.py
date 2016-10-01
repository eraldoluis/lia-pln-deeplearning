from data.Context import Window
from data.FeatureGenerator import FeatureGenerator


class WordWindowGenerator(FeatureGenerator):
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
        self.__window = Window(windowSize)
        self.__embedding = embedding
        self.__filters = filters

        self.__startPaddingIdx, self.__endPaddingIdx = Window.checkPadding(startPadding, endPadding, embedding)

    def generate(self, rawData):
        '''
        Receives a list of tokens and returns window of words.

        :type rawData: list[basestring]
        :param rawData: a list of tokens
        :return:
        '''
        tokens = rawData
        tknIdxs = []

        for token in tokens:
            for f in self.__filters:
                token = f.filter(token,rawData)
            print token,
            print " ",

            tknIdxs.append(self.__embedding.put(token))

        print ""

        x = []
        windowGen = self.__window.buildWindows(tknIdxs, self.__startPaddingIdx, self.__endPaddingIdx)

        for window in windowGen:
            x.append(window)

        return x
