'''
Created on Dec 22, 2019

@author: Dirk Toewe
'''

import numpy as np
import numpy.linalg as la

# import pyximport; pyximport.install( pyimport=True, language_level=3 )
from srrqr.srrqr_l2r import srrqr_l2r
from srrqr.srrqr_bin import srrqr_bin


def test( srrqr ):
  def rank_deficient():
    for _ in range(512):
      m,n = np.random.randint(1,256, size=2)
      A = np.random.rand(m,n)*4 - 2
  
      # RANDOM RANK-DEFICIENT
      U,S,V = la.svd(A, full_matrices=False)
      rank = np.random.randint(0,min(m,n)+1)
      S[rank:] = 0
      S = np.diag(S)
  
      A = U @ S @ V
      A.flags.writeable = False

      yield A, rank

  def matrices():
    for _ in range(512):
      # RANDOM
      m,n = np.random.randint(1,256, size=2)

      A = np.random.rand(m,n)*4 - 2
      A = A.astype(np.float64)
      A.flags.writeable = False
      yield A

#       # BAD CASE FOR (WEAK) RRQR
#       n = np.random.randint(1,256)
#      
#       ang = np.random.rand() * np.pi/2
#       c = np.cos(ang)
#       s = np.sin(ang)
#   
#       S = np.diag(c**np.arange(n))
#       K = np.triu(np.full((n,n), -s))
#       np.fill_diagonal(K, 1)
#   
#       A = S @ K
#       A = A.astype(np.float64)
#       A.flags.writeable = False
#   
#       yield A


  for A in matrices():
    m,n = A.shape
    print('SHAPE: (%3d,%3d)' % (m,n))

    Q,R,P, rank, _  = srrqr(A)
    assert rank <= min(m,n)

    assert np.allclose( np.eye(m), Q.T @ Q   )
    assert np.allclose( np.eye(m), Q   @ Q.T )
    assert np.allclose( 0, np.tril(R,-1) )
    assert np.allclose(np.sort(P), np.arange(n))

    assert np.allclose( A[:,P], Q @ R )
    assert np.allclose( A[:,P], Q[:,:rank] @ R[:rank] )


  for A,RANK in rank_deficient():
    m,n = A.shape
    print('shape: (%3d,%3d)' % (m,n))

    Q,R,P, rank, _ = srrqr(A)
    assert rank <= min(m,n)
    assert rank == RANK, '%d != %d' % (rank,RANK)

    assert np.allclose( np.eye(m), Q.T @ Q   )
    assert np.allclose( np.eye(m), Q   @ Q.T )
    assert np.allclose( 0, np.tril(R,-1) )
    assert np.allclose(np.sort(P), np.arange(n))

    assert np.allclose( A[:,P], Q @ R )
    assert np.allclose( A[:,P], Q[:,:rank] @ R[:rank] )


if __name__ == '__main__':
  test(srrqr_l2r)
  test(srrqr_bin)
