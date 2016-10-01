from data.FeatureGenerator import FeatureGenerator


class LabelGenerator(FeatureGenerator):
    '''
    Receives a list of tags or labels and returns a list of integers which represent these labels or tags.
    '''

    def __init__(self,labelLexicon):
        '''
        :type labelLexicon: data.Lexicon.Lexicon
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

        y = []

        for label in labels:
            i = self.__labelLexicon.put(label)

            if i == -1:
                raise Exception("Label doesn't exist: %s" % label)

            y.append(i)

        return y
