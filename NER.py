#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
#import sys, os
import time

from Data.FeatureFactory import *
from Datum import *
from WindowModelByWord import *
from WindowModelBasic import *
from CharWNN import *
import cPickle as pickle
from Evaluate.EvaluatePrecisionRecallF1 import EvaluatePrecisionRecallF1
from Evaluate.EvaluateAccuracy import EvaluateAccuracy

def main():
    
    
    parser = argparse.ArgumentParser();
    
    parser.add_argument('--withCharwnn',dest='withCharwnn',action='store_true',
                       help='Set training with character embeddings')
    
    parser.add_argument('--train', dest='train', action='store',
                       help='Training File Path',required=True)
    
    parser.add_argument('--test', dest='test', action='store',
                       help='TypeTest File Path',required=True)
    
    parser.add_argument('--numepochs', dest='numepochs', action='store', type=int, required=True,
                       help='Number of epochs: how many iterations over the training set.')
    
    parser.add_argument('--hiddenLayerSize', dest='hiddenSize', action='store', type=int,
                       help='The number of neurons in the hidden layer',default=100)
    
    parser.add_argument('--convolutionalLayerSize', dest='convSize', action='store', type=int,
                       help='The number of neurons in the convolutional layer',default=50)
    
    parser.add_argument('--wordWindowSize', dest='wordWindowSize', action='store', type=int,
                       help='The size of words for the wordsWindow',default=5)
    
    parser.add_argument('--charWindowSize', dest='charWindowSize', action='store', type=int,
                       help='The size of char for the charsWindow',default=5)

    parser.add_argument('--batchSize', dest='batchSize', action='store', type=int,
                       help='The size of the batch in the train',default=1)
    
    parser.add_argument('--lr', dest='lr', action='store', type=float , default=0.0075,
                       help='The value of the learning rate')

    parser.add_argument('--c', dest='c', action='store', type=float , default=0.0,
                       help='The larger C is the more the regularization pushes the weights of all our parameters to zero.')
    
    parser.add_argument('--wordVecSize', dest='wordVecSize', action='store', type=int,
                       help='Word vector size',default=50)
    
    parser.add_argument('--charVecSize', dest='charVecSize', action='store', type=int,
                       help='Char vector size',default=10)

    parser.add_argument('--wordVocab', dest='wordVocab', action='store',
                       help='Word Vocabulary File Path')
    
    parser.add_argument('--charVocab', dest='charVocab', action='store',
                       help='Char Vocabulary File Path')
    
    parser.add_argument('--wordVectors', dest='wordVectors', action='store',
                       help='Word Vectors File Path')
    
    parser.add_argument('--charVectors', dest='charVectors', action='store',
                       help='Char Vectors File Path')
    
    parser.add_argument('--saveModel', dest='saveModel', action='store',
                       help='The file path where the model will be saved')
    
    parser.add_argument('--loadModel', dest='loadModel', action='store',
                       help='The file path where the model is stored')
    
    args = parser.parse_args();
    
     
    t0 = time.time()
    if args.withCharwnn:
        charVecSize = args.charVecSize
    else:
        charVecSize = 0
    
    featureFactory = FeatureFactory(args.wordVecSize,charVecSize)
    
    print 'Loading word dictionary...'
    if args.wordVocab is not None or args.wordVectors is not None:
        
        if args.wordVocab is None or args.wordVectors is None:
            raise ValueError("The word vocabulary file path and wordVector file path has to be set together")
        
        featureFactory.readWordVectors(args.wordVectors, args.wordVocab)
        
        addUnknownWord = False
    else:
        addUnknownWord = True
        
    print 'Loading char dictionary...'
    if args.charVocab is not None or args.charVectors is not None:
        
        if args.charVocab is None or args.charVectors is None:
            raise ValueError("The char vocabulary file path and charVector file path has to be set together")
        
        featureFactory.readCharVectors(args.charVectors, args.charVocab)
        
        addUnknownChar = False
    else:
        addUnknownChar = True

    print 'Loading train data...'
    if args.withCharwnn:
        trainData = featureFactory.readDataWithChar(args.train,addUnknownWord,addUnknownChar)
    else:    
        trainData = featureFactory.readData(args.train,addUnknownWord)
   
    numClasses = featureFactory.getNumberOfLabel()
    
    
    if args.loadModel is not None:
        print 'Loading model in ' + args.loadModel + ' ...'
        f = open(args.loadModel, "rb");
        model = pickle.load(f)
        f.close()
    else:
        if args.withCharwnn==False:
            model = WindowModelByWord(featureFactory.getLexicon(),featureFactory.getWordVector(), 
                            args.wordWindowSize, args.hiddenSize, args.lr,numClasses,args.numepochs,args.batchSize, args.c,None);
            print 'Training...'
            model.train(trainData[0],trainData[1]);
        else:
            
            charModel = CharWNN(trainData[2],trainData[3],featureFactory.getCharcon(),featureFactory.getCharVector(),
                           args.charWindowSize,args.wordWindowSize, args.convSize, args.lr, numClasses, args.numepochs,args.batchSize, args.c);

            wordModel = WindowModelByWord(featureFactory.getLexicon(),featureFactory.getWordVector(), 
                            args.wordWindowSize, args.hiddenSize, args.lr,numClasses,args.numepochs,args.batchSize, args.c,charModel);
    
            wordModel.train(trainData[0],trainData[1])
        
        
        if args.saveModel is not None:
            print 'Saving Model...'
            f = open(args.saveModel, "wb");
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            
            print 'Model save with sucess in ' + args.saveModel,

    t1 = time.time()
    print ("Train time: %s seconds" % (str(t1 - t0)))

    print 'Loading test data...'
    
    
    
    if args.withCharwnn==False:
        testData = featureFactory.readTestData(args.test)
        print 'Testing...'
        predicts = model.predict(testData[0]);
        
    else:
        testData = featureFactory.readTestData(args.test)
                
        print 'Testing...'
        predicts = wordModel.predict(testData[0]);
        
    #eval = EvaluatePrecisionRecallF1(numClasses)
    eval = EvaluateAccuracy()
     
    
    #eval.evaluate(predicts,testData[1]);
    
    
    eval.evaluateWithPrint(predicts,testData[1])

    t2 = time.time()
    print ("TypeTest  time: %s seconds" % (str(t2 - t1)))
    print ("Total time: %s seconds" % (str(t2 - t0)))
    
    
if __name__ == '__main__':
    main()
    

