# Script for Figure 6 in AKSV, "A Compressed Sensing View of Unsupervised Text Embeddings," ICLR'18.
# sh sparse_recovery/scripts-AKSV2018/Figure6.sh

for EMBEDDING in Amazon_SN_ Amazon_GloVe_ Rademacher_; do
  echo $EMBEDDING
  for METHOD in SSH BP LASSO+ LASSO OMP+ OMP; do
    echo $METHOD
    python -W ignore sparse_recovery/word_embeddings.py SST 100 $METHOD $EMBEDDING'50' $EMBEDDING'100' $EMBEDDING'200' $EMBEDDING'400'
  done
  echo ''
done
