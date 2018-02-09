# Script for Figure 7 in AKSV, "A Compressed Sensing View of Unsupervised Text Embeddings," ICLR'18.
# sh sparse_recovery/scripts-AKSV2018/Figure7.sh

for EMBEDDING in Amazon_SN_ Amazon_GloVe_ Rademacher_; do
  echo $EMBEDDING
  for METHOD in SSH BP OMP+ OMP; do
    echo $METHOD
    python -W ignore sparse_recovery/word_embeddings.py IMDB 100 $METHOD $EMBEDDING'200' $EMBEDDING'400' $EMBEDDING'800' $EMBEDDING'1600'
  done
  echo ''
done
