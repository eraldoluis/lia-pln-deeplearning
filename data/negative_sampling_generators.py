import math

from data import FeatureGenerator
from data.FeatureGenerator import FeatureGenerator
from data.WordWindowGenerator import WordWindowGenerator

class NegativeSamplingLabelGenerator(FeatureGenerator):
    """
    Create the label for each example of the negative sampling.
    This class needs to be synchronized with NegativeSamplingWindowGenerator
    """
    def __init__(self, noiseRate):
        self.__noiseRate = noiseRate


    def generate(self, sequence):
        """
        The sequence is a vector, which your size is equal to the number of tokens in sentence.
        For each elemente if this sequence, we are insert one label 1 and noiseRate labels 0.

        Thi

        :param sequence:
        :return:
        """
        labels = []

        for _ in sequence:
            labels.append(1)

            for _ in xrange(self.__noiseRate):
                labels.append(0)

        return labels



class NegativeSamplingWindowGenerator(WordWindowGenerator):
    """
    Create the correct window and the noise windows. These scond type of windows
    are created by changing the center word of the window by other words.
    The quantity of noise words is defined by the parameter noiseRate and these words
    are drawn following a distribution.
    """

    def __init__(self, noiseRate, sampler, windowSize, lexicon, filters, startPadding, endPadding=None):
        super(NegativeSamplingWindowGenerator, self).__init__(windowSize, lexicon, filters, startPadding, endPadding)

        self.__noiseRate = noiseRate
        self.__sampler = sampler

    def generate(self, sequence):
        windowToReturn = []

        for window in super(NegativeSamplingWindowGenerator, self).generate(sequence):
            centerWord = int(len(window) / 2)

            windowToReturn.append(window)

            for _ in xrange(self.__noiseRate):
                noiseWindow = list(window)
                noiseTokenId = self.__sampler.sample()[0]

                noiseWindow[centerWord] = noiseTokenId

                windowToReturn.append(noiseWindow)

        return windowToReturn
