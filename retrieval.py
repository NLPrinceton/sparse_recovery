import numpy as np
np.seterr(all='raise')
from numpy.linalg import inv
from scipy import sparse as sp
from sklearn.linear_model import Lasso
from sparse_recovery.solvers import *
from text_embedding.documents import *
from text_embedding.features import *


# NOTE: LASSO with default noise parameter (alpha=1.0) recovers poorly. Highly recommended to use alpha <= 1E-3.
def recover_features(A, B, method='BP', verbose=False, **kwargs):
  '''recovers sparse feature signals from measurements given the design matrix
  Args:
    A: design matrix of size (d, V)
    B: matrix of measurements of size (n_samples, d)
    method: recovery algorithm to use; must be 'BP', 'LASSO', 'MP', or 'OMP'
    verbose: display progress
    kwargs: kwargs to pass to solvers; for LASSO 'fit_intercept=False' will be set automatically
  Returns:
    matrix (in CSR format) of recovered signals of size (n_samples, V)
  '''

  if not method == 'LASSO':
    _, V = A.shape
    n_samples, _ = B.shape
    output = sp.lil_matrix((n_samples, V))
  if method == 'OMP':
    kwargs['orthogonal'] = True
    method = 'MP'

  if method == 'BP':
    if not kwargs.get('positive', False):
      kwargs['ATinvAAT'] = A.T.dot(inv(A.dot(A.T)))
    for i, b in enumerate(B):
      if verbose:
        write('\rRecovering Signal '+str(i+1)+'/'+str(B.shape[0])+' Using Basis Pursuit')
      output[i] = np.maximum(np.round(BP(A, b, **kwargs)), 0.0)
  elif method == 'LASSO':
    kwargs['fit_intercept'] = False
    if verbose:
      write('\rRecovering '+str(B.shape[0])+' Signals Using LASSO')
    return Lasso(**kwargs).fit(A, B.T).sparse_coef_.maximum(0.0).rint()
  elif method == 'MP':
    for i, b in enumerate(B):
      write('\rRecovering Signal '+str(i+1)+'/'+str(B.shape[0])+' Using Greedy Pursuit')
      output[i] = np.maximum(np.round(MP(A, b, **kwargs)), 0.0)
  else:
    raise(NotImplementedError)

  return output.tocsr()


# NOTE: valid when both predicted and truth take nonnegative values
# NOTE: if truth takes integer values round predicted values to nearest integers
def precision(predicted, truth):
  '''computes precision of predicted features
  Args:
    predicted: array of size (n_samples, n_features)
    truth: array of size (n_samples, n_features)
  Returns:
    vector of length n_samples containing the precision of each sample
  '''

  assert not (predicted<0).sum(), "matrix of predicted values must be nonnegative"
  assert not (truth<0).sum(), "matrix of true values must be nonnegative"
  predicted = sp.csr_matrix(predicted)
  truth = sp.csr_matrix(truth)

  pslice = np.array(np.greater(predicted.sum(1), 0))[:,0]
  precision = np.zeros(pslice.shape)
  if sum(pslice):
    diff = predicted[pslice] - truth[pslice]
    precision[pslice] = np.array(1 - diff.multiply(diff>0).sum(1) / predicted[pslice].sum(1))[:,0]
  return precision


# NOTE: valid when both predicted and truth take nonnegative values
# NOTE: if truth takes integer values round predicted values to nearest integers
def recall(predicted, truth):
  '''computes recall of predicted features
  Args:
    predicted: array of size (n_samples, n_features)
    truth: array of size (n_samples, n_features)
  Returns:
    vector of length n_samples containing the recall of each sample
  '''

  assert not (predicted<0).sum(), "matrix of predicted values must be nonnegative"
  assert not (truth<0).sum(), "matrix of true values must be nonnegative"
  predicted = sp.csr_matrix(predicted)
  truth = sp.csr_matrix(truth)

  pslice = np.array(np.greater(predicted.sum(1), 0))[:,0]
  recall = np.zeros(pslice.shape)
  if sum(pslice):
    diff = truth[pslice] - predicted[pslice]
    recall[pslice] = np.array(1 - diff.multiply(diff>0).sum(1) / truth[pslice].sum(1))[:,0]
  return recall


# NOTE: valid when both predicted and truth take nonnegative values
# NOTE: if truth takes integer values round predicted values to nearest integers
def f1score(predicted, truth):
  '''computes F1-score of predicted features
  Args:
    predicted: array of size (n_samples, n_features)
    truth: array of size (n_samples, n_features)
  Returns:
    vector of length n_samples containing the F1-score of each sample
  '''

  pr = precision(predicted, truth)
  re = recall(predicted, truth)
  pslice = np.greater(np.greater(pr, 0) + np.greater(re, 0), 0)
  f1 = np.zeros(pslice.shape)
  if sum(pslice):
    f1[pslice] = 2*pr[pslice]*re[pslice]/(pr[pslice]+re[pslice])
  return f1


def information_preservation(documents, f2v, m=40, shp=False, sorted_features=sorted, verbose=False, random_state=0, **kwargs):
  '''evaluates recovery of sparse document featurizations from linear measurements
  Args:
    documents: list of featurized documents (lists of hashable features)
    f2v: dict mapping features to vectors
    m: sample size
    shp: check Supporting Hyperplane Property instead of recovery
    sorted_features: function that sorts the features
    verbose: display recovery statistics before returning
    random_state: seed for subsampling
    kwargs: passed to recover_features or SHP
  Returns:
    if shp returns featurizations (sparse CSR matrix with m rows) and Boolean vector of length m indicating SHP; otherwise returns featurizations and recovered featurizations
  '''

  vocabulary = {feat for doc in documents for feat in doc}
  featset = vocabulary.intersection(f2v.keys())
  docs = [doc for doc in documents if featset.issuperset(doc)]
  assert len(docs) >= m, "number of documents with all features represented is less than the sample size"
  featlist = sorted(featset)
  np.random.seed(random_state)
  original = docs2bofs(np.random.choice(docs, size=m, replace=False), vocabulary=featlist)
  A = np.vstack(f2v[feat] for feat in featlist)

  if shp:
    S = np.array([type(SHP(x, A.T, **kwargs)) == np.ndarray for i, x in enumerate(original) if not verbose or write('\rChecking SHP of Document '+str(i+1)+'/'+str(m))])
    if verbose:
      write('\r'+str(np.sum(S))+'/'+str(m)+' Documents Satisfy SHP'+20*' '+'\n')
    return original, S

  recovered = recover_features(A.T, original.dot(A), verbose=verbose, **kwargs)
  if verbose:
    pr = np.mean(precision(recovered, original))
    re = np.mean(recall(recovered, original))
    f1 = f1score(recovered, original)
    write('\r'+str((f1==1.0).sum())+'/'+str(m)+' Documents Recovered Perfectly'+20*' '+'\n')
    write('Precision: '+str(pr)+', Recall: '+str(re)+', F1-Score: '+str(np.mean(f1))+'\n')
  return original, recovered
