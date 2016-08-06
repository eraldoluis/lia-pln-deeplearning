#############################################
Exemplo de uso do postagger:
##############################################
**É necessário deletar a primeira e última linha do arquivo de saída

nohup java -mx200m -cp  stanford-postagger.jar edu.stanford.nlp.process.PTBTokenizer -preserveLines exemplo.txt > exemplo_tokenizado.txt 2>&1 &

###############################################
Exemplo de uso de word2vec:
###############################################

nohup ./word2vec -train exemplo_tokenizado.txt -output exemplo_wordvec.txt -window 3 -threads 20 -iter 25 -min-count 2 -size 30 -sg 1 -negative 5 -seed 1443901331  2>&1 &








