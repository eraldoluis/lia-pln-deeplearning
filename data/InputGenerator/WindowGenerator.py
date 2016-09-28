from data.InputGenerator.FeatureGenerator import FeatureGenerator


class WindowGenerator(FeatureGenerator):
    """
    Generate window of tokens from each word of a list. This list can be a line.
    """

    def __init__(self, windowSize, embedding, filters, startPadding, endPadding=None):
        """
        :type windowSize: int
        :param windowSize: the size of window

        :type embedding: data.Embedding.Embedding
        :param embedding:

        :type filters: list[data.Filters.Filter]
        :param filters:

        :param startPadding: Object that will be place when the initial limit of a list is exceeded

        :param endPadding: Object that will be place when the end limit a list is exceeded.
            If this parameter is None, so the endPadding has the same value of startPadding
        """
        self.__embedding = embedding
        self.__filters = filters
        self.__startPaddingIdx = None
        self.__endPaddingIdx = None

        self.setPadding(startPadding, endPadding)

        if windowSize % 2 == 0:
            raise Exception("The given window size (%d) is even but should be odd." % windowSize)
        self.__windowSize = windowSize

    def setPadding(self, startPadding, endPadding):
        """
        Verify if the start padding and end padding exist in lexicon or embedding.

        :param startPadding: Object that will be place when the initial limit of a list is exceeded

        :param endPadding: Object that will be place when the end limit a list is exceeded.
            If this parameter is None, so the endPadding has the same value of startPadding

        :return: the index of start and end padding in lexicon
        """

        embedding = self.__embedding

        if not embedding.exist(startPadding):
            if embedding.isReadOnly():
                raise Exception("Start padding symbol does not exist")

            self.__startPaddingIdx = embedding.put(startPadding)
        else:
            self.__startPaddingIdx = embedding.getLexiconIndex(startPadding)

        if endPadding is not None:
            if not embedding.exist(endPadding):
                if embedding.isReadOnly():
                    raise Exception("End padding symbol does not exist")

                self.__endPaddingIdx = embedding.put(endPadding)
            else:
                self.__endPaddingIdx = embedding.getLexiconIndex(endPadding)
        else:
            self.__endPaddingIdx = None

    def generate(self, tokens):
        """
        Receives a list of tokens and returns window of words.

        :type tokens: list[basestring]
        :param tokens: list of tokens
        :return:
        """
        tknIdxs = []

        for token in tokens:
            for f in self.__filters:
                token = f.filter(token)
            tknIdxs.append(self.__embedding.put(token))

        return self.__buildWindows(tknIdxs)

    def __buildWindows(self, tokens):
        """
        Receives a list of objects and creates the windows of this list.

        :type tokens: []
        :param tokens: list of objects that will be used to build windows. Objects can be anything even lists.

        :param startPadding: Object that will be place when the initial limit of list is exceeded

        :param endPadding: Object that will be place when the end limit of objs is exceeded.
            When this parameter is null, so the endPadding has the same value of startPadding

        :return Returns a generator from yield
        """

        endPadding = self.__endPaddingIdx
        startPadding = self.__startPaddingIdx
        winSize = self.__windowSize

        numTkns = len(tokens)
        contextSize = (winSize - 1) / 2

        # List of token indexes expanded with padding symbols.
        tokensWithPadding = [startPadding] * contextSize + tokens + [endPadding] * contextSize;

        windows = []
        for idx in xrange(numTkns):
            windows.append(tokensWithPadding[idx:idx + winSize])

        return windows
