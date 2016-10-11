from data.FeatureGenerator import FeatureGenerator
from data.Window import Window


class WordWindowGenerator(FeatureGenerator):
    """
    Generate window of words from each word of a list.
    This list can be a line.
    """

    def __init__(self, windowSize, lexicon, filters,
                 startPadding, endPadding=None):
        """
        :type windowSize: int
        :param windowSize: the size of window

        :type lexicon: data.Lexicon.Lexicon
        :param lexicon:

        :type filters: list[DataOperation.Filters.Filter]
        :param filters:

        :param startPadding: Object that will be place when the initial limit of a list is exceeded

        :param endPadding: Object that will be place when the end limit a list is exceeded.
            If this parameter is None, so the endPadding has the same value of startPadding
        """
        self.__window = Window(lexicon, windowSize, startPadding, endPadding)
        self.__lexicon = lexicon
        self.__filters = filters

    def generate(self, sequence):
        """
        Receives a sequence of tokens and returns a sequence of token windows.

        :type sequence: list[basestring]
        :param sequence: sequence of tokens
        :return: a sequence of token windows.
        """
        tknIdxs = []
        for token in sequence:
            for f in self.__filters:
                token = f.filter(token, sequence)
            tknIdxs.append(self.__lexicon.put(token))

        return self.__window.buildWindows(tknIdxs)
