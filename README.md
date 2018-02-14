# sparse_recovery

This module provides solvers and utility functions for several problems in sparse recovery/compressed sensing. The file <tt>solvers.py</tt>, which depends only on NumPy/SciPy and can be used with Python 2 or 3, provides methods to compute the following:
  * Basis Pursuit (BP), including Nonnegative BP (BP+); the BP solver is ported from the l1-MAGIC MATLAB package [1]; the BP+ solver is based on the same primal-dual interior point method [2].
  * Orthogonal Matching Pursuit (OMP), including Nonnegative OMP (OMP+).
  * Supporting Hyperplane Property (SHP), a property that guarantees recovery of a signal x from linear measurements Ax via BP+ [3].
  
The feature retrieval files (<tt>retrieval.py</tt>, <tt>word_embeddings.py</tt>) additionally require scikit-learn and text_embedding. These files are used by scripts in the directory <tt>scripts-AKSV2018</tt> to recreate the results in [3].

If you find this code useful please cite the following:
  
    @inproceedings{arora2018sensing,
      title={A Compressed Sensing View of Unsupervised Text Embeddings, Bag-of-n-Grams, and LSTMs},
      author={Arora, Sanjeev and Khodak, Mikhail and Saunshi, Nikunj and Vodrahalli, Kiran},
      booktitle={To Appear in the Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2018}
    }
    
# References:
[1] Candès & Romberg, "l1-MAGIC: Recovery of Sparse Signals via Convex Programming," *Technical Report*, 2005.

[2] Boyd & Vandenberghe, "Chapter 11: Interior-point Methods," *Convex Optimization*, 2004.

[3] Arora et al., "A Compressed Sensing View of Unsupervised Text Embedding, Bag-of-n-Grams, and LSTMS," *ICLR*, 2018.
