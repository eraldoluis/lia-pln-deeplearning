from DataOperation.InputGenerator.FeatureGenerator import FeatureGenerator


class LabelGenerator(FeatureGenerator):
    '''
    Receives a list of tags or labels and returns a list of integers which represent these labels or tags.
    '''

    def __init__(self,labelLexicon):
        '''
        :type labelLexicon: DataOperation.Lexicon.Lexicon
        :param labelLexicon:
        '''
        self.__labelLexicon = labelLexicon

    def generate(self, labels):
        '''
        Returns a list of integers which represent these labels or tags.

        :type labels: list[basestring]
        :param labels:

        :return: li
        '''

        y = [ self.__labelLexicon.put(label) for label in labels]

        return y
