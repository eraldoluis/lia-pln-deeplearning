#!/bin/bash

LR=0.01
VOCAB=$None
TIPO=$None
TRAIN=$None
TEST=$None
PREDICTION=$None
MODEL=$None
OUT="saida.out"
ALG=$None
OOUV=$None
OOSV=$None
CROSS=$None
ITER=30

for i in "$@"
do
case $i in
    -train=*)
    TRAIN="${i#*=}"
    shift # past argument=value
    ;;
    -test=*)
    TEST="${i#*=}"
    shift # past argument=value
    ;;
    -prediction=*)
    PREDICTION="${i#*=}"
    shift # past argument=value
    ;;
    -model=*)
    MODEL="${i#*=}"
    shift # past argument=value
    ;;	
    -out=*)
    OUT="${i#*=}"
    shift # past argument=value
    ;;		
    -iter=*)
    ITER="${i#*=}"
    shift # past argument=value
    ;;
    -vocab=*)
    VOCAB="${i#*=}"
    shift # past argument=value
    ;;
    -cross=*)
    CROSS="--crossvalidation --kfold ${i#*=}"
    shift # past argument=value
    ;;				
    -lr=*)
    LR="${i#*=}"
    shift # past argument=value
    ;;		
    --pos)
    ALG="Postag.py"
    shift # past argument=value
    ;;
    --as)
    ALG="SentimentAnalysis.py"
    shift # past argument=value
    ;;
    --wnn)
    TIPO="--senlayerwithact --norm_coef=0.5"
    shift # past argument=value
    ;;
    --charwnn)
    TIPO="--withCharwnn --charwnnwithact --senlayerwithact"
    shift # past argument=value
    ;;
    --oouv)
    OOUV="--testoouv"
    shift # past argument=value
    ;;
    --oosv)
    OOSV="--testoosv"
    shift # past argument=value
    ;;		
    *)
    UNKNOWN="${i#*=}"
    echo "Warning: Unknown Option = ${UNKNOWN}"        # unknown option
    ;;
esac
done

nohup python -u ../${ALG} --train=${TRAIN} --test=${TEST} --saveModel=${MODEL} ${TIPO} ${OOSV} ${OOUV} --savePrediction=${PREDICTION} --numepochs=${ITER} ${CROSS} --lr=${LR} --vocab=${VOCAB} --hiddenlayersize=300 --charConvolutionalLayerSize=50 --wordConvolutionalLayerSize=300 --wordWindowSize=5 --charWindowSize=3 --numperepoch 1  --wordVecSize=30 --charVecSize=5  --maxSizeOfWord=30 --startSymbol '<s>' --endSymbol '</s>' --filters DataOperation.TransformNumberToZeroFilter TransformNumberToZeroFilter DataOperation.RemoveURL RemoveURL DataOperation.RemoveUserName RemoveUserName --lrupdstrategy=divide_epoch --unknownwordstrategy="mean_vector" --mean_size 1000 --seed=1443901331 --charVecsInit='randomAll' --wordVecsInit='zscore' --adagrad > ${OUT} 2>&1 &






