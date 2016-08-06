#!/bin/bash

TRAIN=$None
TEST=$None
PREDICTION=$None
MODEL=$None
OUT="saida.out"
ALG=$None
OOUV=$None
OOSV=$None
ITER=1

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
    --pos)
    ALG="Postag.py"
    shift # past argument=value
    ;;
    --as)
    ALG="SentimentAnalysis.py"
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


nohup python -u ../${ALG} --train=${TRAIN} --test=${TEST} --loadModel=${MODEL} ${OOSV} ${OOUV} --savePrediction=${PREDICTION} --numepochs=${ITER} > ${OUT} 2>&1 &


