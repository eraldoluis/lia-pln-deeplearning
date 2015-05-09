#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time

from DataOperation.MacMophorReader import MacMorphoReader
from DataOperation.WordVector import WordVector
from Evaluate.EvaluateAccuracy import EvaluateAccuracy
from WindowModelBySentence import WindowModelBySentence
import cPickle as pickle
from DataOperation.Lexicon import Lexicon
from WindowModelByWord import WindowModelByWord


def main():
    
    parser = argparse.ArgumentParser();
    
    parser.add_argument('--train', dest='train', action='store',
                       help='Training File Path',required=True)
    
    parser.add_argument('--test', dest='test', action='store',
                       help='TypeTest File Path',required=True)    
    parser.add_argument('--numepochs', dest='numepochs', action='store', type=int, required=True,
                       help='Number of epochs: how many iterations over the training set.')
    
    algTypeChoices= ["window_word","window_sentence"]
    
    parser.add_argument('--alg', dest='alg', action='store', default="window_sentence", choices=algTypeChoices,
                       help='O algorithm use to train and test')
    
    parser.add_argument('--hiddenlayersize', dest='hiddenSize', action='store', type=int,
                       help='The number of neurons in the hidden layer',default=100)
    
    parser.add_argument('--windowsize', dest='windowSize', action='store', type=int,
                       help='The number of neurons in the hidden layer',default=5)

    parser.add_argument('--batchSize', dest='batchSize', action='store', type=int,
                       help='The size of the batch in the train',default=1)
    
    parser.add_argument('--lr', dest='lr', action='store', type=float , default=0.001,
                       help='The value of the learning rate')

    parser.add_argument('--c', dest='c', action='store', type=float , default=0.0,
                       help='The larger C is the more the regularization pushes the weights of all our parameters to zero.')    
    parser.add_argument('--wordvecsize', dest='wordVecSize', action='store', type=int,
                       help='Word vector size',default=50)

    parser.add_argument('--vocab', dest='vocab', action='store',
                       help='Vocabulary File Path')
    
    parser.add_argument('--wordvectors', dest='wordVectors', action='store',
                       help='word Vectors File Path')
    parser.add_argument('--savemodel', dest='saveModel', action='store',
                       help='The file path where the model will be saved')
    
    parser.add_argument('--loadmodel', dest='loadModel', action='store',
                       help='The file path where the model is stored')
    
    args = parser.parse_args();
    
        
    t0 = time.time()

    print 'Loading dictionary...'

    if args.vocab is not None or args.wordVectors is not None:
        
        if args.vocab is None or args.wordVectors is None:
            raise ValueError("The vocabulary file path and wordVector file path has to be set together")
        
        wordVector = WordVector(args.wordVectors)
        lexicon = Lexicon(args.vocab)
        
        if lexicon.getLen() != wordVector.getLength():
            raise Exception("O número de palavras no vacabulário é diferente do número de palavras do word Vector")
        
        addUnkownWord = False
    else:
        wordVector = WordVector(wordSize=args.wordVecSize)
        lexicon = Lexicon()
        addUnkownWord = True
        
    
    datasetReader = MacMorphoReader()

    print 'Loading train data...'
    
    lexiconOfLabel = Lexicon()
    
    if args.loadModel is not None:
        print 'Loading model in ' + args.loadModel + ' ...'
        f = open(args.loadModel, "rb");
        model = pickle.load(f)
        
        if isinstance(model,WindowModelByWord):
            separeSentence = False
        elif isinstance(model,WindowModelBySentence): 
            separeSentence = True
            
        f.close()
    else:
        if args.alg == algTypeChoices[0]:
            separeSentence = False
            trainData = datasetReader.readData(args.train,lexicon,lexiconOfLabel,wordVector,separeSentence,addUnkownWord)
            
            numClasses = lexiconOfLabel.getLen()
            model = WindowModelByWord(lexicon,wordVector, 
                            args.windowSize, args.hiddenSize, args.lr,numClasses,args.numepochs,args.batchSize, args.c);
        
        elif args.alg == algTypeChoices[1]:
            separeSentence = True
            trainData = datasetReader.readData(args.train,lexicon,lexiconOfLabel,wordVector,separeSentence,addUnkownWord)
            
            numClasses = lexiconOfLabel.getLen()
            model = WindowModelBySentence(lexicon,wordVector, 
                            args.windowSize, args.hiddenSize, args.lr,numClasses,args.numepochs,args.batchSize, args.c)
        
        print 'Training...'
        model.train(trainData[0],trainData[1]);
        
        if args.saveModel is not None:
            print 'Saving Model...'
            f = open(args.saveModel, "wb");
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            
            print 'Model save with sucess in ' + args.saveModel,

    t1 = time.time()
    print ("Train time: %s seconds" % (str(t1 - t0)))

    print 'Loading test data...'
    testData = datasetReader.readData(args.test,lexicon,lexiconOfLabel,wordVector,separeSentence)
    print 'Testing...'
    predicts = model.predict(testData[0]);
    
    numClasses = lexiconOfLabel.getLen()
    
    eval = EvaluateAccuracy(numClasses)
    
    eval.evaluate(predicts,testData[1]);
    

    t2 = time.time()
    print ("TypeTest  time: %s seconds" % (str(t2 - t1)))
    print ("Total time: %s seconds" % (str(t2 - t0)))
    
if __name__ == '__main__':
    main()
    

