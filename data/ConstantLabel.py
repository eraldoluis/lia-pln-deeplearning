from data.FeatureGenerator import FeatureGenerator


class ConstantLabel(FeatureGenerator):
    '''
        This class gives always the same label for a input.
    '''

    def __init__(self, labelLexicon, label):
        '''
        :type labelLexicon: DataOperation.Lexicon.Lexicon
        :param labelLexicon:
        '''
        self.__labelId = labelLexicon.put(label)

    def generate(self, labels):
        '''
        Returns a list of integers which represent these labels or tags.

        :type labels: list[basestring]
        :param labels:

        :return: li
        '''

        y = []

        for label in labels:
            i = self.__labelId

            if i == -1:
                raise Exception("Label doesn't exist: %s" % label)

            y.append(i)

        return y

