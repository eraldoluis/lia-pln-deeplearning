#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time

from DataOperation.MacMophorReader import MacMorphoReader
from DataOperation.WordVector import WordVector
from Evaluate.EvaluateAccuracy import EvaluateAccuracy
from WindowModelBySentence import WindowModelBySentence, NeuralNetworkChoiceEnum
import cPickle as pickle
from DataOperation.Lexicon import Lexicon
from WindowModelByWord import WindowModelByWord
import sys
import numpy
from WindowModelBasic import WindowModelBasic
from Evaluate.EvaluateEveryNumEpoch import EvaluateEveryNumEpoch
from DataOperation.ReaderLexiconAndWordVec import ReaderLexiconAndWordVec
import importlib
from NNet.Util import LearningRateUpdDivideByEpochStrategy,\
    LearningRateUpdNormalStrategy
from Evaluate.EvaluatePercPredictsCorrectNotInWordSet import EvaluatePercPredictsCorrectNotInWordSet


def readVocabAndWord(args):
    if args.vocab == args.wordVectors or args.vocab is not None or args.wordVectors is not None:
        fi = args.wordVectors if args.wordVectors is not None else args.vocab
        lexicon, wordVector = ReaderLexiconAndWordVec().readData(fi)
    else:
        wordVector = WordVector(args.wordVectors)
        lexicon = Lexicon(args.vocab)
    
    return lexicon, wordVector

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
    
    parser.add_argument('--numperepoch', dest='numPerEpoch', action='store', nargs='*', type=int,
                       help="The evaluation on the test corpus will "
                            + "be performed after a certain number of training epoch."
                            + "If the value is an integer, so the evalution will be performed after this value of training epoch "
                            + "If the value is a list of integer, than the evaluation will be performed when the epoch is equal to one of list elements.  ",default=None)

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
    
    parser.add_argument('--startsymbol', dest='startSymbol', action='store', default="<s>",
                       help='The symbol that represents the begin a setence')
    
    parser.add_argument('--endsymbol', dest='endSymbol', action='store',default="</s>",
                       help='The symbol that represents the end a setence')
    
    parser.add_argument('--filters', dest='filters', action='store',default=[],nargs='*',
                       help='The filters which will be apply in the data. You have to pass the module and class name.'+
                       ' Ex: modulename1 classname1 modulename2 classname2')
    
    parser.add_argument('--testoosv', dest='testOOSV', action='store_true',default=False,
                       help='Do the test OOSV')
    
    parser.add_argument('--testoouv', dest='testOOUV', action='store_true',default=False,
                       help='Do the test OOSV')
    
    unkownWordStrategy= ["random","mean_all_vector","word_vocab"]
    
    
    parser.add_argument('--unkownwordstrategy', dest='unkownWordStrategy', action='store',default=unkownWordStrategy[0]
                        ,choices=unkownWordStrategy,
                       help='Choose the strategy use to construct a word vector of a unknown word.'
                       + 'There are three types of strategy: random(generate randomly a word vector) ,'+
                       ' mean_all_vector(generate the word vector form the mean of all words)' +
                       ', word_vocab(use a word vector of one particular word. You have to use the parameter unkownword to set this word)')
    
    parser.add_argument('--unkownword', dest='unkownWord', action='store',default=False,
                       help='The worch which will use the represent a unkown word')
    
    lrStrategyChoices= ["normal","divide_epoch"]
    
    parser.add_argument('--lrupdstrategy', dest='lrUpdStrategy', action='store',default=lrStrategyChoices[0],choices=lrStrategyChoices,
                       help='Set the learning rate update strategy. NORMAL and DIVIDE_EPOCH are the options available')
    
    
    
    networkChoices= ["complete","without_hidden_update_wv" ,"without_update_wv"]
    
    parser.add_argument('--networkChoice', dest='networkChoice', action='store',default=networkChoices[0],choices=networkChoices)
    
    #Todo: delete
#     numpy.random.seed(10)
    try:
        args = parser.parse_args();
        
        print args
    except:
        parser.print_help()
        sys.exit(0)
        
    
    filters = []
    a = 0
    
    args.filters.append('DataOperation.TransformLowerCaseFilter')
    args.filters.append('TransformLowerCaseFilter')
    
    while a < len(args.filters):
        module_ = importlib.import_module(args.filters[a])
        filters.append(getattr(module_, args.filters[a+1])())
        a+=2
        
    t0 = time.time()
 
    datasetReader = MacMorphoReader(False)
    testData = None
    
    if args.testOOUV:
        unkownDataTest = []
    else:
        unkownDataTest = None
        
    if args.loadModel is not None:
        print 'Loading model in ' + args.loadModel + ' ...'
        f = open(args.loadModel, "rb");
        lexicon,lexiconOfLabel,model = pickle.load(f)
        
        if isinstance(model,WindowModelByWord):
            separeSentence = False
        elif isinstance(model,WindowModelBySentence): 
            separeSentence = True
            
        if args.testOOSV:
            lexiconFindInTrain = set()
            datasetReader.readData(args.train,lexicon,lexiconOfLabel, separateSentences= separeSentence,filters=filters,lexiconFindInTrain=lexiconFindInTrain)
            
        if args.testOOUV:
            lexiconWV, wv = readVocabAndWord(args)
            lexiconFindInWV = set([ word for word in lexiconWV.getLexiconDict()])
            
        f.close()
    else:        
        print 'Loading dictionary...'
        
        WindowModelBasic.setStartSymbol(args.startSymbol)
        WindowModelBasic.setEndSymbol(args.endSymbol)
        
        

        if args.vocab is not None or args.wordVectors is not None:
            lexicon, wordVector = readVocabAndWord(args)
            
            if lexicon.isUnknownIndex(lexicon.getLexiconIndex(WindowModelBasic.startSymbolStr)):
                raise Exception("O vocabulário não possui o símbolo de começo\"<s>)\"")
            if lexicon.isUnknownIndex(lexicon.getLexiconIndex(WindowModelBasic.endSymbolStr)):
                raise Exception("O vocabulário não possui o símbolo de final\"<s>)\"")
            if lexicon.getLen() != wordVector.getLength():
                raise Exception("O número de palavras no vacabulário é diferente do número de palavras do word Vector")
            
            if args.testOOUV:
                lexiconFindInWV = set([ word for word in lexicon.getLexiconDict()])
                
            addWordUnkown = False
        else:
            wordVector = WordVector(wordSize=args.wordVecSize)
            lexicon = Lexicon()
            
            lexicon.put(WindowModelBasic.startSymbolStr)
            wordVector.append(None)
            
            lexicon.put(WindowModelBasic.endSymbolStr)
            wordVector.append(None)
            
            if args.testOOUV:
                lexiconFindInWV = set()
            
            addWordUnkown = True
            
        
        args.unkownWordStrategy
        
        unknowName = u'UUUNKKK'
        
        if args.unkownWordStrategy == unkownWordStrategy[0]:
            lexiconIndex = lexicon.put(unknowName)
            lexicon.setUnkownIndex(lexiconIndex)
            
            wordVector.append(None)
        elif args.unkownWordStrategy == unkownWordStrategy[1]:
            unkownWordVector = numpy.mean(numpy.asarray(wordVector.getWordVectors()),0)
            
            lexiconIndex = lexicon.put(unknowName)
            lexicon.setUnkownIndex(lexiconIndex)
            
            wordVector.append(unkownWordVector.tolist())
            
        elif args.unkownWordStrategy == unkownWordStrategy[2]:
            lexiconIndex = lexicon.getLexiconIndex(unicode(args.unkownWord, "utf-8"))
            
            if lexicon.isUnknownIndex(lexiconIndex):
                raise Exception('Unkown Word Value passed not exist in the unsupervised dictionary');
            
            lexicon.setUnkownIndex(lexiconIndex)
            
        lexiconOfLabel = Lexicon()
        
        if args.lrUpdStrategy == lrStrategyChoices[0]:
            learningRateUpdStrategy = LearningRateUpdNormalStrategy()
        elif args.lrUpdStrategy == lrStrategyChoices[1]:
            learningRateUpdStrategy = LearningRateUpdDivideByEpochStrategy()
        
        lexiconFindInTrain = set() if args.testOOSV else None
        
        if args.alg == algTypeChoices[0]:
            separeSentence = False
            print 'Loading train data...'
            trainData = datasetReader.readData(args.train,lexicon,lexiconOfLabel,wordVector,separeSentence,addWordUnkown,filters,lexiconFindInTrain)
            
            numClasses = lexiconOfLabel.getLen()
            model = WindowModelByWord(lexicon,wordVector, 
                            args.windowSize, args.hiddenSize, args.lr,numClasses,args.numepochs,args.batchSize, args.c,learningRateUpdStrategy);
        
        elif args.alg == algTypeChoices[1]:
            separeSentence = True
            print 'Loading train data...'
            trainData = datasetReader.readData(args.train,lexicon,lexiconOfLabel,wordVector,separeSentence,addWordUnkown,filters,lexiconFindInTrain)
            
            if args.networkChoice == networkChoices[0]:
                networkChoice = NeuralNetworkChoiceEnum.COMPLETE
            elif args.networkChoice == networkChoices[1]:
                networkChoice = NeuralNetworkChoiceEnum.WITHOUT_HIDDEN_LAYER_AND_UPD_WV
            elif args.networkChoice == networkChoices[2]:
                networkChoice = NeuralNetworkChoiceEnum.WITHOUT_UPD_WV
            else:
                networkChoice = NeuralNetworkChoiceEnum.COMPLETE
            
            numClasses = lexiconOfLabel.getLen()
            model = WindowModelBySentence(lexicon,wordVector, 
                            args.windowSize, args.hiddenSize, args.lr,numClasses,args.numepochs,args.batchSize, args.c,learningRateUpdStrategy,networkChoice)
                        
        if args.numPerEpoch is not None and len(args.numPerEpoch) != 0 :
            print 'Loading test data...'
            testData = datasetReader.readTestData(args.test,lexicon,lexiconOfLabel,separeSentence,filters,unkownDataTest)
            
            evalListener = EvaluateEveryNumEpoch(args.numepochs,args.numPerEpoch,EvaluateAccuracy(),model,testData[0],testData[1])
            
            model.addListener(evalListener)
            
        print 'Training...'
        model.train(trainData[0],trainData[1]);
        
        if args.saveModel is not None:
            print 'Saving Model...'
            f = open(args.saveModel, "wb");
            pickle.dump([lexicon,lexiconOfLabel,model], f, pickle.HIGHEST_PROTOCOL)
            f.close()
            
            print 'Model save with sucess in ' + args.saveModel

    t1 = time.time()
    print ("Train time: %s seconds" % (str(t1 - t0)))
    
    print 'Loading test data...'
    
    if testData is None:          
        testData = datasetReader.readTestData(args.test,lexicon,lexiconOfLabel,separeSentence,filters,unkownDataTest)

    
    print 'Testing...'
    predicts = model.predict(testData[0]);
    
    eval = EvaluateAccuracy()
    acc = eval.evaluateWithPrint(predicts,testData[1]);
    
    if args.testOOSV:
        oosv = EvaluatePercPredictsCorrectNotInWordSet(lexicon,lexiconFindInTrain,'OOSV')
        oosv.evaluateWithPrint(predicts, testData[1], testData[0],unkownDataTest)
    
    if args.testOOUV:
        oouv = EvaluatePercPredictsCorrectNotInWordSet(lexicon,lexiconFindInWV,'OOUV')
        oouv.evaluateWithPrint(predicts, testData[1], testData[0],unkownDataTest)
    

    t2 = time.time()
    print ("Test  time: %s seconds" % (str(t2 - t1)))
    print ("Total time: %s seconds" % (str(t2 - t0)))
    
if __name__ == '__main__':
    main()
    

