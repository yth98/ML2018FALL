cat w2* > word2vec_2.model
cat w3* > word2vec_3.model
cat w4* > word2vec_4.model
python3 hw4-ensemble.py $1 $2 $3
