####################################################################
Treinando e testando modelo com script
####################################################################
--------------------------------------------------------------------
Treinar um modelo:
--------------------------------------------------------------------
Exemplo:

./train_model.sh -train=dataset/ctrls/ctrls.tokenized.txt_x_6_folds/train_1.txt -test=dataset/ctrls/ctrls.tokenized.txt_x_6_folds/test_1.txt -prediction=predicao.txt -model=models/treino_wnn.model -out=saida.out --as --wnn -iter=2 -vocab=dataset/ctrls/twi_pt_win5_min_20_siz_30_sg_1_neg_5.txt -lr=0.01 --wnn

tal que: 

-train: arquivo que foi utilizado para o treino do modelo (obrigatório)
-test: arquivo de teste (obrigatório)
-prediction: arquivo onde deseja salvar a predição do teste
-model: arquivo que deseja salvar o modelo
-out: arquivo onde deseja imprimir a saída (com o valor da acurácia)
-iter: número de iterações (default 30)
-vocab: vocabulário para o treino
-cross: número de folds para validação cruzada (a validação é ativada com este parâmetro)
-lr: taxa de aprendizado (Default 0.01)

** Os parâmetros de ativação:
É necessário escolher entre:
--as : o algoritmo que será executado é a análise de sentimentos
--pos : o algoritmo que será executado é o POS tagging

E entre:
--wnn: o modelo wnn será usado
--charwnn: o modelo charwnn será usado

Outros parâmetros:
--oouv : calcula acurácia sobre as palavras fora do vocabulário de treino
--oosv : calcula acurácia sobre as palavras fora do treino


-------------------------------------------------------------------
Testar um modelo treinado: 
-------------------------------------------------------------------

./test_model.sh -test=dataset/ctrls/ctrls.tokenized.txt_x_6_folds/test_1.txt -prediction=predicao.txt -model=models/ctrls/ctrls_wnn.model -out=saida.out --as

tal que: 

-train: arquivo que foi utilizado para o treino do modelo (requerido quando usa --oouv ou --oosv)
-test: arquivo de teste (obrigatório)
-prediction: arquivo onde deseja salvar a predição do teste
-model: o modelo que deseja testar (obrigatório)
-out: arquivo onde deseja imprimir a saída (com o valor da acurácia)
-iter: número de iterações 

 Os parâmetros de ativação:

--as : o algoritmo que será executado é a análise de sentimentos
--pos : o algoritmo que será executado é o POS tagging
--oouv : calcula acurácia sobre as palavras fora do vocabulário de treino
--oosv : calcula acurácia sobre as palavras fora do treino

####################################################################
Treinando e testando modelo sem script
####################################################################

Treino:
**Exemplo para Charwnn

nohup python -u ../SentimentAnalysis.py --train=dataset/ctrls/ctrls.tokenized.txt_x_6_folds/train_1.txt --test=dataset/ctrls/ctrls.tokenized.txt_x_6_folds/test_1.txt --numepochs 2 --hiddenlayersize=300 --charConvolutionalLayerSize=50 --wordConvolutionalLayerSize=300 --wordWindowSize=5 --charWindowSize=3 --numperepoch 1 --lr=0.01 --wordVecSize=30 --charVecSize=5  --maxSizeOfWord=30 --startSymbol '<s>' --endSymbol '</s>' --filters DataOperation.TransformNumberToZeroFilter TransformNumberToZeroFilter DataOperation.RemoveURL RemoveURL DataOperation.RemoveUserName RemoveUserName --testoosv --testoouv --lrupdstrategy=divide_epoch --unknownwordstrategy="mean_vector" --mean_size 1000 --seed=1443901331 --charVecsInit='randomAll' --wordVecsInit='zscore' --adagrad --saveModel=models/modelo_charwnn.model --vocab=dataset/ctrls/twi_pt_win5_min_20_siz_30_sg_1_neg_5.txt --norm_coef=1.0 --withCharwnn --charwnnwithact --senlayerwithact > testando.txt 2>&1 &


