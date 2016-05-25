#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import logging.config

from DataOperation.TokenLabelReader import TokenLabelReader
from DataOperation.WordVector import WordVector
from Evaluate.EvaluateAccuracy import EvaluateAccuracy
from WindowModelBySentence import WindowModelBySentence, NeuralNetworkChoiceEnum
from NNet.EmbeddingConvolutionalLayer import EmbeddingConvolutionalLayer
import cPickle as pickle
from DataOperation.Lexicon import Lexicon
from WindowModelByWord import WindowModelByWord
import sys
import numpy
from WindowModelBasic import WindowModelBasic
from Evaluate.EvaluateEveryNumEpoch import EvaluateEveryNumEpoch
from DataOperation.ReaderLexiconAndWordVec import ReaderLexiconAndWordVec
import importlib
from NNet.Util import LearningRateUpdDivideByEpochStrategy, \
    LearningRateUpdNormalStrategy
from Evaluate.EvaluatePercPredictsCorrectNotInWordSet import EvaluatePercPredictsCorrectNotInWordSet
import random
import os

def readVocabAndWord(args):
    # Este if só foi criado para permitir que DLIDPostag possa reusar o run do postag
    if isinstance(args.vocab, Lexicon) and isinstance(args.wordVectors, (WordVector,list)): 
        return args.vocab, args.wordVectors
    
    if args.vocab == args.wordVectors or args.vocab is not None or args.wordVectors is not None:
        fi = args.wordVectors if args.wordVectors is not None else args.vocab
        lexicon, wordVector = ReaderLexiconAndWordVec().readData(fi)
    else:
        wordVector = WordVector(args.wordVectors)
        lexicon = Lexicon(args.vocab)
    
    return lexicon, wordVector

def run(args):
    
    filters = []
    a = 0
    args.filters.append('DataOperation.TransformLowerCaseFilter')
    args.filters.append('TransformLowerCaseFilter')
    while a < len(args.filters):
        print "Usando o filtro: " + args.filters[a] + " " + args.filters[a + 1]
        module_ = importlib.import_module(args.filters[a])
        filters.append(getattr(module_, args.filters[a + 1])())
        a += 2
    
    t0 = time.time()
    datasetReader = TokenLabelReader(args.fileWithFeatures, args.tokenLabelSeparator)
    testData = None
    if args.testOOUV:
        unknownDataTest = []
    else:
        unknownDataTest = None
    # charVars = [charcon, charVector, charIndexesOfLexiconRaw, numCharsOfLexiconRaw]
    charVars = [None, None, {}, []]

    if args.loadModel:
        print 'Loading model from ' + args.loadModel + ' ...'
        f = open(args.loadModel, "rb")
        lexicon, lexiconOfLabel, lexiconRaw, model, charVars = pickle.load(f)
        f.close()

        if isinstance(model, WindowModelByWord):
            separeSentence = False
        elif isinstance(model, WindowModelBySentence):
            separeSentence = True
        if model.charModel == None:
            print "The loaded model does not include a char embedding"
            args.withCharwnn = False
        else:
            print "The loaded model includes a char embedding"
            args.withCharwnn = True

        model.setTestValues = True
        
        if args.testOOSV:
            lexiconFindInTrain = set()
            unkownData = []
            # datasetReader.readData(args.train,lexicon,lexiconOfLabel, separateSentences=separateSentence,filters=filters,lexiconFindInTrain=lexiconFindInTrain)
            datasetReader.readData(args.train, lexicon, lexiconOfLabel, 
                                   lexiconRaw, separateSentences=separeSentence, 
                                   withCharwnn=args.withCharwnn, 
                                   charVars=charVars, filters=filters, 
                                   setWordsInDataSet=lexiconFindInTrain,unknownDataTestCharIdxs=unkownData)
            
            if model.charModel is not None:
                model.charModel.updateAllCharIndexes(unkownData)
        if args.testOOUV:
            if args.vocab is not None or args.wordVectors is not None:
                lexiconWV, _ = readVocabAndWord(args)
                lexiconFindInWV = set([word for word in lexiconWV.getLexiconDict()])
            else:
                lexiconFindInWV = set([word for word in lexicon.getLexiconDict()])
                
            # lexiconFindInWV = set()
            # datasetReader.readData(args.train,lexiconWV,lexiconOfLabel, lexiconRaw, separateSentences=separeSentence,withCharwnn=args.withCharwnn,
            #                       charVars=charVars,filters=filters,setWordsInDataSet=lexiconFindInWV)
    else:
        print 'Loading dictionary...'
        
        WindowModelBasic.setStartSymbol(args.startSymbol)
        WindowModelBasic.setEndSymbol(args.endSymbol)
        
        lexiconRaw = Lexicon()
        charcon = Lexicon()
        charIndexesOfLexiconRaw = {}
        
        numCharsOfLexiconRaw = []
        charVector = WordVector(wordSize=args.charVecSize, mode=args.charVecsInit)
        # charVector = WordVector(wordSize=args.charVecSize)
        
        if args.vocab is not None or args.wordVectors is not None:
            lexicon, wordVector = readVocabAndWord(args)
            
            if lexicon.isUnknownIndex(lexicon.getLexiconIndex(WindowModelBasic.startSymbolStr)):
                raise Exception("O vocabulário não possui o símbolo de começo\"<s>)\"")
            if lexicon.isUnknownIndex(lexicon.getLexiconIndex(WindowModelBasic.endSymbolStr)):
                raise Exception("O vocabulário não possui o símbolo de final\"<\s>)\"")
#             if lexicon.getLen() != wordVector.getLength():
#                 raise Exception("O número de palavras no vacabulário é diferente do número de palavras do word Vector")
            if isinstance(wordVector, WordVector):
                wvs = [wordVector]
            else:
                wvs = wordVector
                    
            for wv in wvs:
                if lexicon.getLen() != wv.getLength():
                    raise Exception("O número de palavras no vacabulário é diferente do número de palavras do word Vector")
            

            if args.testOOUV:
                lexiconFindInWV = set([word for word in lexicon.getLexiconDict()])
            addWordUnknown = False
        else:
            wordVector = WordVector(wordSize=args.wordVecSize, mode=args.wordVecsInit)
            lexicon = Lexicon()
            
            lexicon.put(WindowModelBasic.startSymbolStr)
            wordVector.append(None)
            
            if WindowModelBasic.startSymbolStr != WindowModelBasic.endSymbolStr:
                lexicon.put(WindowModelBasic.endSymbolStr)
                wordVector.append(None)
            if args.testOOUV:
                lexiconFindInWV = set()
            addWordUnknown = True
        
        unknownNameDefault = u'UUUNKKK'
        
        if isinstance(wordVector, WordVector):
            wordVector = [wordVector]

        if args.unknownWordStrategy == "random":
            if lexicon.isWordExist(unknownNameDefault):
                raise Exception(unknownNameDefault + u' already exists in the vocabulary.')
            
            lexiconIndex = lexicon.put(unknownNameDefault)
            lexicon.setUnknownIndex(lexiconIndex)
#             wordVector.append(None)
            for wv in wordVector:
                wv.append(None) 
            
        
        elif args.unknownWordStrategy == "mean_vector":
            if lexicon.isWordExist(unknownNameDefault):
                raise Exception(unknownNameDefault + u' already exists in the vocabulary.')
            if args.meanSize < 1 and args.meanSize > 0:
                mean_size = int(wordVector.getLength() * args.meanSize)
            else:
                mean_size = int(args.meanSize)
            
            lexiconIndex = lexicon.put(unknownNameDefault)
            lexicon.setUnknownIndex(lexiconIndex)
#             wordVector.append(unknownWordVector.tolist())
            for wv in wordVector:
                unknownWordVector = numpy.mean(numpy.asarray(wv.getWordVectors()[wv.getLength() - mean_size:]), 0)
                wv.append(unknownWordVector.tolist())
            

        elif args.unknownWordStrategy == "word_vocab":
            lexiconIndex = lexicon.getLexiconIndex(unicode(args.unknownWord, "utf-8"))
            if lexicon.isUnknownIndex(lexiconIndex):
                raise Exception('Unknown Word Value passed does not exist in the unsupervised dictionary')
            lexicon.setUnknownIndex(lexiconIndex)
        
        else:
            raise Exception('Unknown Word Value passed does not exist in the unsupervised dictionary')
        
        if args.withCharwnn:
            
            idx = lexiconRaw.put(WindowModelBasic.startSymbolStr)
            idx2 = charcon.put(WindowModelBasic.startSymbolStr)
            charVector.append(None)
            numCharsOfLexiconRaw.append(1)
            charIndexesOfLexiconRaw[idx] = [idx2]
            
            if WindowModelBasic.startSymbolStr != WindowModelBasic.endSymbolStr:
                idx = lexiconRaw.put(WindowModelBasic.endSymbolStr)
                idx2 = charcon.put(WindowModelBasic.endSymbolStr)
                charVector.append(None)
                numCharsOfLexiconRaw.append(1)
                charIndexesOfLexiconRaw[idx] = [idx2]
            
            if lexiconRaw.getLexiconIndex(unknownNameDefault) is lexiconRaw.getUnknownIndex():
                idx = lexiconRaw.put(unknownNameDefault)
                lexiconRaw.setUnknownIndex(idx)
                idx2 = charcon.put(unknownNameDefault)
                charcon.setUnknownIndex(idx2)
                numCharsOfLexiconRaw.append(1)
                charIndexesOfLexiconRaw[idx] = [idx2]
                charVector.append(None)
            # charVars = [charcon,charVector, charIndexesOfLexiconRaw, numCharsOfLexiconRaw]
            charVars = [charcon, charVector, charIndexesOfLexiconRaw, numCharsOfLexiconRaw]
        
        lexiconOfLabel = Lexicon()
        charModel = None

        if args.lrUpdStrategy == "normal":
            learningRateUpdStrategy = LearningRateUpdNormalStrategy()
        elif args.lrUpdStrategy == "divide_epoch":
            learningRateUpdStrategy = LearningRateUpdDivideByEpochStrategy()
        
        lexiconFindInTrain = set() if args.testOOSV else None
        
        
        if args.networkChoice == "complete":
            networkChoice = NeuralNetworkChoiceEnum.COMPLETE
        elif args.networkChoice == "without_hidden_update_wv":
            networkChoice = NeuralNetworkChoiceEnum.WITHOUT_HIDDEN_LAYER_AND_UPD_WV
        elif args.networkChoice == "without_update_wv":
            networkChoice = NeuralNetworkChoiceEnum.WITHOUT_UPD_WV
        else:
            networkChoice = NeuralNetworkChoiceEnum.COMPLETE
        
        if args.alg == "window_word":
            separeSentence = False
            print 'Loading train data...'
            trainData = datasetReader.readData(args.train, lexicon,
                                               lexiconOfLabel, lexiconRaw,
                                               wordVector, separeSentence,
                                               addWordUnknown, args.withCharwnn,
                                               charVars, True, filters,
                                               lexiconFindInTrain)
            
            numClasses = lexiconOfLabel.getLen()
            
            if args.withCharwnn:
                if args.charVecsInit == 'randomAll':
                    charVars[1].startAllRandom()
                if args.charVecsUpdStrategy == 'min_max' or args.charVecsInit == 'min_max':
                    charVars[1].minMax(args.norm_coef)
                elif args.charVecsUpdStrategy == 'z_score' or args.charVecsInit == 'z_score':
                    charVars[1].zScore(args.norm_coef)

                # TODO: o tamanho da representação de caracteres está fixa (20).
                # Precisamos colocar isso como argumento do programa.
                charModel = EmbeddingConvolutionalLayer(charVars[0], charVars[1], 
                                                        charVars[2], 20, 
                                                        args.charWindowSize, 
                                                        args.wordWindowSize,
                                                        args.convSize, 
                                                        numClasses, args.c, 
                                                        learningRateUpdStrategy, 
                                                        separeSentence, 
                                                        args.charwnnWithAct, 
                                                        args.charVecsUpdStrategy, 
                                                        args.networkAct, 
                                                        args.norm_coef)
            
            if args.wordVecsInit == 'randomAll':
                wordVector.startAllRandom()
            if args.wordVecsUpdStrategy == 'min_max' or args.wordVecsInit == 'min_max':
                wordVector.minMax(args.norm_coef)
            elif args.wordVecsUpdStrategy == 'z_score' or args.wordVecsInit == 'z_score':
                wordVector.zScore(args.norm_coef)
            
            model = WindowModelByWord(lexicon, wordVector, args.wordWindowSize,
                                      args.hiddenSize, args.lr, numClasses,
                                      args.numepochs, args.batchSize, args.c,
                                      charModel, learningRateUpdStrategy,
                                      args.wordVecsUpdStrategy, args.networkAct,
                                      args.norm_coef, not args.noStructGrad,
                                      adaGrad=args.adaGrad,
                                      randomizeInput=not args.notRandomizeInput,
                                      embeddingNotUpdate = args.nonupdatewv,
                                      choiceNetwork=networkChoice,
                                      loadHiddenWeights=args.loadHiddenWeights)
        
        elif args.alg == "window_sentence":
            separeSentence = True
            
            print 'Loading train data...'
            trainData = datasetReader.readData(args.train, lexicon,
                                               lexiconOfLabel, lexiconRaw,
                                               wordVector, separeSentence,
                                               addWordUnknown, args.withCharwnn,
                                               charVars, True, filters,
                                               lexiconFindInTrain)

            numClasses = lexiconOfLabel.getLen()
            
            if args.withCharwnn:
                if args.charVecsInit == 'randomAll':
                    charVars[1].startAllRandom()
                if args.charVecsUpdStrategy == 'min_max' or args.charVecsInit == 'min_max':
                    charVars[1].minMax(args.norm_coef)
                elif args.charVecsUpdStrategy == 'z_score' or args.charVecsInit == 'z_score':
                    charVars[1].zScore(args.norm_coef)
                charModel = EmbeddingConvolutionalLayer(charVars[0], charVars[1], charVars[2], charVars[3], args.charWindowSize, args.wordWindowSize,
                    args.convSize, numClasses, args.c, learningRateUpdStrategy, separeSentence, args.charwnnWithAct, args.charVecsUpdStrategy, args.networkAct, args.norm_coef)
            
            if args.wordVecsInit == 'randomAll':
                wordVector.startAllRandom()
            if args.wordVecsUpdStrategy == 'min_max' or args.wordVecsInit == 'min_max':
                wordVector.minMax(args.norm_coef)
            elif args.wordVecsUpdStrategy == 'z_score' or args.wordVecsInit == 'z_score':
                wordVector.zScore(args.norm_coef)
            
            model = WindowModelBySentence(lexicon, wordVector, args.wordWindowSize, 
                                          args.hiddenSize, args.lr, numClasses, 
                                          args.numepochs, args.batchSize, args.c, 
                                          charModel, learningRateUpdStrategy, 
                                          args.wordVecsUpdStrategy, networkChoice, 
                                          args.networkAct, args.norm_coef)
        
        if args.numPerEpoch is not None and len(args.numPerEpoch) != 0:
            print 'Loading test data...'
            unknownDataTestCharIdxs = []
            
            testData = datasetReader.readTestData(args.test, lexicon, lexiconOfLabel, lexiconRaw, separeSentence, False, args.withCharwnn, charVars, False, filters, unknownDataTest, unknownDataTestCharIdxs)
            
            evalListener = EvaluateEveryNumEpoch(args.numepochs, args.numPerEpoch, EvaluateAccuracy(), model, testData[0], testData[1], testData[2], unknownDataTestCharIdxs)
            
            model.addListener(evalListener)
        
        print 'Training...'
        model.train(trainData[0], trainData[1], trainData[2])
        
        if args.saveModel is not None:
            print 'Saving Model...'
            f = open(args.saveModel, "wb")
            
            # When the test data is loaded by the script, but no one predict is done, 
            #   so it's necessary to update all char indexes, because the lexiconRaw is going to have data,
            #    that is not in AllCharWindowIndexes.
            if charModel is not None and len(charModel.AllCharWindowIndexes) != lexiconRaw.getLen():
                charModel.updateAllCharIndexes(unknownDataTestCharIdxs)
            
            pickle.dump([lexicon, lexiconOfLabel, lexiconRaw, model, charVars], f, pickle.HIGHEST_PROTOCOL)
            
            f.close()
            print 'Model save with sucess in ' + args.saveModel
    
    t1 = time.time()
    
    print "Train time: %s seconds" % (str(t1 - t0))
    print 'Loading test data...'
    
    if testData is None:
        unknownDataTestCharIdxs = []
        testData = datasetReader.readTestData(args.test, lexicon, lexiconOfLabel, lexiconRaw, separeSentence, False, args.withCharwnn, charVars, False, filters, unknownDataTest, unknownDataTestCharIdxs)
    
    print 'Testing...'
    
    predicts = model.predict(testData[0], testData[2], unknownDataTestCharIdxs)
    predicts_y_given_x = model.predict(testData[0], testData[2], unknownDataTestCharIdxs)
    
    
    if args.savePrediction is not None:
            print 'Saving Prediction...'
            f = open(args.savePrediction, "wb")
            pickle.dump([predicts_y_given_x,predicts, testData[1],lexiconOfLabel], f, pickle.HIGHEST_PROTOCOL)
            f.close()
            print 'Prediction save with sucess in ' + args.savePrediction
    
    
    evalue = EvaluateAccuracy()
    evalue.evaluateWithPrint(predicts, testData[1])

    if args.testOOSV:
        oosv = EvaluatePercPredictsCorrectNotInWordSet(lexicon, lexiconFindInTrain, 'OOSV')
        oosv.evaluateWithPrint(predicts, testData[1], testData[0], unknownDataTest)
    
    if args.testOOUV:
        oouv = EvaluatePercPredictsCorrectNotInWordSet(lexicon, lexiconFindInWV, 'OOUV')
        oouv.evaluateWithPrint(predicts, testData[1], testData[0], unknownDataTest)
    
    t2 = time.time()
    
    print "Test  time: %s seconds" % (str(t2 - t1))
    print "Total time: %s seconds" % (str(t2 - t0))

def main():
    
    parser = argparse.ArgumentParser();
    
    parser.add_argument('--train', dest='train', action='store',
                       help='Training File Path', required=True)
    
    parser.add_argument('--test', dest='test', action='store',
                       help='TypeTest File Path', required=True)
        
    parser.add_argument('--numepochs', dest='numepochs', action='store', type=int, required=True,
                       help='Number of epochs: how many iterations over the training set.')
    
    parser.add_argument('--withCharwnn', dest='withCharwnn', action='store_true',
                       help='Set training with character embeddings')
    
    parser.add_argument('--tokenLabelSeparator', dest='tokenLabelSeparator', action='store', required=False, default="_",
                            help="Specify the character that is being used to separate the token from the label in the dataset.")
    
    parser.add_argument('--alg', dest='alg', action='store',
                        default="window_sentence",
                        choices=["window_word", "window_sentence"],
                        help='The type of algorithm to train and test')
    
    parser.add_argument('--hiddenlayersize', dest='hiddenSize', action='store', type=int,
                       help='The number of neurons in the hidden layer', default=300)
    
    parser.add_argument('--convolutionalLayerSize', dest='convSize', action='store', type=int,
                       help='The number of neurons in the convolutional layer', default=50)
    
    parser.add_argument('--wordWindowSize', dest='wordWindowSize', action='store', type=int,
                       help='The size of words for the wordsWindow', default=5)
    
    parser.add_argument('--charWindowSize', dest='charWindowSize', action='store', type=int,
                       help='The size of char for the charsWindow', default=5)
    
    parser.add_argument('--numperepoch', dest='numPerEpoch', action='store', nargs='*', type=int,
                       help="The evaluation on the test corpus will "
                            + "be performed after a certain number of training epoch."
                            + "If the value is an integer, so the evalution will be performed after this value of training epoch "
                            + "If the value is a list of integer, than the evaluation will be performed when the epoch is equal to one of list elements.  ", default=None)

    parser.add_argument('--batchSize', dest='batchSize', action='store', type=int,
                       help='The size of the batch in the train', default=1)
    
    parser.add_argument('--lr', dest='lr', action='store', type=float , default=0.0075,
                       help='The value of the learning rate')

    parser.add_argument('--c', dest='c', action='store', type=float , default=0.0,
                       help='The larger C is the more the regularization pushes the weights of all our parameters to zero.')    
    
    parser.add_argument('--wordVecSize', dest='wordVecSize', action='store', type=int,
                       help='Word vector size', default=100)
    
    parser.add_argument('--charVecSize', dest='charVecSize', action='store', type=int,
                       help='Char vector size', default=10)

    parser.add_argument('--vocab', dest='vocab', action='store',
                       help='Vocabulary File Path')
    
    parser.add_argument('--wordVectors', dest='wordVectors', action='store',
                       help='word Vectors File Path')
    
    parser.add_argument('--saveModel', dest='saveModel', action='store',
                       help='The file path where the model will be saved')
    
    parser.add_argument('--loadModel', dest='loadModel', action='store',
                       help='The file path where the model is stored')
    
    parser.add_argument('--startSymbol', dest='startSymbol', action='store', default="<s>",
                       help='The symbol that represents the beginning of a setence')
    
    parser.add_argument('--endSymbol', dest='endSymbol', action='store', default="</s>",
                       help='The symbol that represents the ending of a setence')
    
    parser.add_argument('--filters', dest='filters', action='store', default=[], nargs='*',
                       help='The filters which will be applied to the data. You have to pass the module and class name.' + 
                       ' Ex: modulename1 classname1 modulename2 classname2')
    
    parser.add_argument('--testoosv', dest='testOOSV', action='store_true', default=False,
                       help='Do the test OOSV')
    
    parser.add_argument('--testoouv', dest='testOOUV', action='store_true', default=False,
                       help='Do the test OOUV')

    parser.add_argument('--unknownwordstrategy', dest='unknownWordStrategy',
                        action='store',
                        default="random",
                        choices=["random", "mean_vector", "word_vocab"],
                        help='Choose the strategy that will be used for ' + 
                        'constructing a word vector of a unknown word. There ' + 
                        'are three types of strategy: random(generate randomly ' + 
                        'a word vector), mean_all_vector(generate the word ' + 
                        'vector from the mean of all words), word_vocab(use' + 
                        ' a word vector of one particular word. You have to' + 
                        ' use the parameter unknownword to set this word)')
    
    parser.add_argument('--unknownword', dest='unknownWord', action='store', default=False,
                       help='The word which will be used to represent the unknown word')

    parser.add_argument('--lrupdstrategy', dest='lrUpdStrategy', action='store',
                        default="normal", choices=["normal", "divide_epoch"],
                       help='Set the learning rate update strategy. ' + 
                       'NORMAL and DIVIDE_EPOCH are the options available')
        
    parser.add_argument('--filewithfeatures', dest='fileWithFeatures', action='store_true',
                       help='Set that the training e testing files have features')
    
    vecsInitChoices = ["randomAll", "random", "zeros", "z_score", "min_max"]
    
    parser.add_argument('--charVecsInit', dest='charVecsInit', action='store', default=vecsInitChoices[1], choices=vecsInitChoices,
                       help='Set the way to initialize the char vectors. RANDOM, RANDOMALL, ZEROS, Z_SCORE and MIN_MAX are the options available')
    
    parser.add_argument('--wordVecsInit', dest='wordVecsInit', action='store', default=vecsInitChoices[1], choices=vecsInitChoices,
                       help='Set the way to initialize the char vectors. RANDOM, RANDOMALL, ZEROS, Z_SCORE and MIN_MAX are the options available')
    
    parser.add_argument('--charwnnwithact', dest='charwnnWithAct', action='store_true',
                       help='Set training with character embeddings')
    
    parser.add_argument('--networkChoice', dest='networkChoice', action='store',
                        default="complete",
                        choices=["complete", "without_hidden_update_wv" , "without_update_wv"])
    
    parser.add_argument('--networkAct', dest='networkAct', action='store',
                        default="tanh",
                        choices=["tanh", "hard_tanh", "sigmoid", "hard_sigmoid", "ultra_fast_sigmoid"])
    
    parser.add_argument('--charNetAct', dest='charNetAct', action='store', 
                        default="tanh", 
                        choices=["tanh","hard_tanh","sigmoid","hard_sigmoid"])
    
    parser.add_argument('--mean_size', dest='meanSize', action='store', type=float, default=1.0,
                       help='The number of the least used words in the train for unknown word' 
                       + 'Number between 0 and 1 for percentage, number > 1 for literal number to make the mean and negative for mean_all')
    
    vecsUpStrategyChoices = ["normal", "min_max", "z_score"]

    parser.add_argument('--wordvecsupdstrategy', dest='wordVecsUpdStrategy', action='store', default=vecsUpStrategyChoices[0], choices=vecsUpStrategyChoices,
                       help='Set the word vectors update strategy. NORMAL, MIN_MAX and Z_SCORE are the options available')
    
    parser.add_argument('--charvecsupdstrategy', dest='charVecsUpdStrategy', action='store', default=vecsUpStrategyChoices[0], choices=vecsUpStrategyChoices,
                       help='Set the char vectors update strategy. NORMAL, MIN_MAX and Z_SCORE are the options available')

    parser.add_argument('--norm_coef', dest='norm_coef', action='store', type=float, default=1.0,
                       help='The coefficient that will be multiplied to the normalized vectors')

    parser.add_argument('--seed', dest='seed', action='store', type=long,
                       help='', default=None)
    
    parser.add_argument('--nostructgrad', dest='noStructGrad', action='store_true',
                       help='Disable structured gradients (in embedding layers, ' + 
                            'for instance), i.e., use only ordinary gradient')
    
    parser.add_argument('--adagrad', dest='adaGrad', action='store_true',
                       help='Activate AdaGrad updates.')
    
    parser.add_argument('--savePrediction', dest='savePrediction', action='store',
                       help='The file path where the prediction will be saved')
    
    parser.add_argument('--nonupdatewv',dest='nonupdatewv', action='store', nargs='*', type=int,default=[],
                        help='Receive word embedding indexes which is not to be updated. The index begin with 0.')
    
    parser.add_argument('--notRandomizeInput', dest='notRandomizeInput', action='store_true',
                       help='The file path where the prediction will be saved')

    parser.add_argument('--loadHiddenWeights', dest="loadHiddenWeights", action='store')
    
    #parser.add_argument('--saveSolution', dest='saveSolution', action='store',
    #                  help='The file path where the prediction will be saved')
    
    logger = logging.getLogger("Logger")
    formatter = logging.Formatter('[%(asctime)s]\t%(message)s')
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.setLevel(logging.INFO)
    
    try:
        args = parser.parse_args();
        print args
        
    except:
        parser.print_help()
        sys.exit(0)
    
    
    # print "using" + os.environ['OMP_NUM_THREADS'] + " threads"
    
    
    if args.seed != None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)

    run(args)

if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
        
    logging.config.fileConfig(os.path.join(path,'logging.conf'))
    main()
