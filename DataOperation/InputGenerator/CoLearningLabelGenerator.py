from DataOperation.InputGenerator.FeatureGenerator import FeatureGenerator


class CoLearningLabelGenerator(FeatureGenerator):
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

        y = []

        for label in labels:
            if label:
                i = self.__labelLexicon.put(label)

                if i < 0:
                    raise Exception("Trying to read a label which doesn't exist in dataset.")

                y.append(i)
            else:
                y.append(-1)

        return y
