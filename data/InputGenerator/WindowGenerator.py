from data import Context
from data.InputGenerator.FeatureGenerator import FeatureGenerator


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

        :type embedding: data.Embedding.Embedding
        :param embedding:

        :type filters: list[data.Filters.Filter]
        :param filters:

        :param startPadding: Object that will be place when the initial limit of a list is exceeded

        :param endPadding: Object that will be place when the end limit a list is exceeded.
            If this parameter is None, so the endPadding has the same value of startPadding
        '''
        self.__window = Context.Window(windowSize)
        self.__embedding = embedding
        self.__filters = filters

        self.__startPaddingIdx, self.__endPaddingIdx = self.checkPadding(startPadding, endPadding, embedding)

    def generate(self, tokens):
        '''
        Receives a list of tokens and returns window of words.

        :type tokens: list[basestring]
        :param tokens: list of tokens
        :return:
        '''
        tknIdxs = []

        for token in tokens:
            for f in self.__filters:
                token = f.filter(token)

            tknIdxs.append(self.__embedding.put(token))

        x = []
        windowGen = self.__window.buildWindows(tknIdxs,
                                               self.__startPaddingIdx,
                                               self.__endPaddingIdx)

        for window in windowGen:
            x.append(window)

        return x

    def checkPadding(self, startPadding, endPadding, embedding):
        '''
        Verify if the start padding and end padding exist in lexicon or embedding.

        :param startPadding: Object that will be place when the initial limit of a list is exceeded

        :param endPadding: Object that will be place when the end limit a list is exceeded.
            If this parameter is None, so the endPadding has the same value of startPadding

        :param embedding: data.Embedding.Embedding

        :return: the index of start and end padding in lexicon
        '''

        if not embedding.exist(startPadding):
            if embedding.isReadOnly():
                raise Exception("Start Padding doens't exist")

            startPaddingIdx = embedding.put(startPadding)
        else:
            startPaddingIdx = embedding.getLexiconIndex(startPadding)
        if endPadding is not None:
            if not embedding.exist(endPadding):
                if embedding.isReadOnly():
                    raise Exception("End Padding doens't exist")

                endPaddingIdx = embedding.put(endPadding)
            else:
                endPaddingIdx = embedding.getLexiconIndex(endPadding)
        else:
            endPaddingIdx = None

        return startPaddingIdx, endPaddingIdx
