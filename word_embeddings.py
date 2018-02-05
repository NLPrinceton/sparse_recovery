import sys
import numpy as np
from sparse_recovery.retrieval import *
from text_embedding.documents import *
from text_embedding.vectors import *


TASKMAP.update({'imdb': imdb, 'sst': sst})
VECTORFILES.update(dict((('Amazon', 'GloVe', dim), '/n/fs/nlpdatasets/AmazonProductData/amazon_glove'+str(dim)+'.txt') for dim in [50, 100, 200, 400, 800, 1600]))
VECTORFILES.update(dict((('Amazon', 'SN', dim), '/n/fs/nlpdatasets/AmazonProductData/amazon_randwalk'+str(dim)+'.txt') for dim in [50, 100, 200, 400, 800, 1600]))


if __name__ == '__main__':

  dataset = sys.argv[1].lower()
  m = int(sys.argv[2])
  method = sys.argv[3]
  kwargs = {'m': m, 'verbose': True}
  if method.lower() == 'ssh':
    kwargs['ssh'] = True
  elif method[-1] == '+':
    kwargs['positive'] = True
    kwargs['method'] = method[:-1]
  else:
    kwargs['method'] = method
  if method[:5].lower() == 'lasso':
    kwargs['alpha'] = 1E-3

  docs = tokenize(doc.lower() for doc in TASKMAP[dataset]('train')[0])
  vocab = {word for doc in docs for word in doc}
  for embedding in sys.argv[4:]:
    try:
      corpus, objective, dimension = embedding.split('_')
      w2v = vocab2vecs(vocab, corpus=corpus, objective=objective, dimension=int(dimension))
      write(' '.join(('Testing Recovery of', str(dataset.upper()), 'BoW from Compression by', dimension+'-Dimensional', corpus, objective, 'Vectors\n')))
    except ValueError:
      random, dimension = embedding.split('_')
      np.random.seed(0)
      w2v = vocab2vecs(vocab, random=random, dimension=int(dimension))
      write(' '.join(('Testing Recovery of', str(dataset.upper()), 'BoW from Compression by', dimension+'-Dimensional', random, 'Vectors\n')))
    information_preservation(docs, w2v, **kwargs)
