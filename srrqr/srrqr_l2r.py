'''
Created on Dec 22, 2019

@author: Dirk Toewe
'''

import numpy as np
import numpy.linalg as  la
import scipy.linalg as sla


def srrqr_l2r( M ):
  '''
  Computes strong rank-revaling QR decompositin using algorithm 4 as described in:

    Ming Gu, Stanley C. Eisenstat,
   "EFFICIENT ALGORITHMS FOR COMPUTING A STRONG RANK-REVEALING QR FACTORIZATION"
    https://math.berkeley.edu/~mgu/MA273/Strong_RRQR.pdf

  The key idea of the SRRQR is that at each step of the QR decomposition columns are swapped
  between the left half (that is already triangular) and the right half, if the column swap
  sufficiently increases the determinant of the left half (and thus decreasing the determinant
  of the lower right quadrant).

  The algorithm tries every potential rank from left (0) to right (n), which is why we
  call this algorithm the left-to-right (L2R) variant.

  Gu and Eisenstat showed that a matrix W can be computed that, for each column swap (i,k+j),
  contains the factor by which the derminant increases. This matrix can be updated with O(n*m)
  iterations during each step of the QR elimination. Thus the SRRQR is only by a constant factor
  slower than the conventional RRQR.
  '''
  DEBUG = False

  M = np.array(M, dtype=np.float64, copy=True)
  M.flags.writeable = False

  m,n = M.shape
  f = 1.01

  assert f >= 1

  Q  = np.eye(m, dtype=np.float64)
  R  = M.copy()
  P  = np.arange(n, dtype=np.int64)

  R_norm = la.norm(R)
  if 0 == R_norm:
    return Q,R,P, 0, 0

  # At each iteration step, the following equation holds true:
  #   M = Q @ R
  #
  # R has three non-zero quadrants:
  #       ┏       ┓
  #       ┃ A ┊ B ┃
  #   R = ┃┈┈┈┼┈┈┈┃
  #       ┃   ┊ C ┃
  #       ┗       ┛
  # Where A is an upper triangular matrix with shape [k,k].
  # C is not (yet) triangular.

  # TODO:
  #   The split k at which norm(C,'fro') is
  #   approximately zero could be found via
  #   binary search.

  AB = np.zeros_like(R) # <- keeps track of inv(A) and A\B (= inv(A) @ B)

  # keeps track of column norms of C
  v = la.norm(R, axis=0, keepdims=True)

  rank = min(m,n)
  k,K, = 0,rank

  def pivot( j ):
    R [:,[k,j]] = R [:,[j,k]]
    AB[:,[k,j]] = AB[:,[j,k]]
    P [  [k,j]] = P [  [j,k]]

  def eliminate(k):
    '''
    Eliminates column k using Householder reflections.
    '''
    u = R[k:,k].copy()
    u[0] += ((u[0] > 0)*2 - 1) * la.norm(u)

    if 0 == u[0]:
      nonlocal rank
      rank = k
      return False

    u /= np.abs(u).max() # <- make underflow-safe (as numpy doesn't seem to guarantee that...)
    u *= np.sqrt(2) / la.norm(u)
    u  = u.reshape(-1,1)
    R[k:,:] -=            u @(u.T @ R[k:,:])
    Q[:,k:] -= (Q[:,k:] @ u)@ u.T
    R[k+1:,k] = 0
    return True

  def update():
    nonlocal k,v
    # update inv(A)
    AB[ k,  k]  = -1
    AB[:k+1,k] /= -R[k,k]

    assert k < K
    k += 1

    # update A \ B (= inv(A) @ B)
    AB[:k,k:] += AB[:k,k-1:k] @ R[k-1:k,k:]

    # recompute column norms v
    v = la.norm(R[k:,k:], axis=0, keepdims=True)

  def downdate():
    nonlocal k
    # downdate A \ B (= inv(A) @ B)
    AB[:k,k:] -= AB[:k,k-1:k] @ R[k-1:k,k:]

    k -= 1 # <- backtrack and let loop eliminate newly swapped-in column

    # downdate inv(A)
    AB[:k+1,k ] *= -R[k,k]
    assert np.allclose(-1, AB[k,k])   # ◀─┬─ If downgraded correctly, row k should be [-1, 0, ..., 0], see upgrade above
    assert np.allclose( 0, AB[k,k+1:])# ◀─╯
    AB[ k,  k:]  = 0 # <- remove cancellation errors from row k

  # If norm(C,'fro') <= ztol, C is considered to be zero.
  # It uses max_abs(R) as scaling factor, which is always
  # less than or equal to norm(M,'fro') and it should
  # therefore be a safely low threshold.
  ztol = np.finfo(R.dtype).eps * max(m,n) * R_norm

  n_swaps = 0

  while k < K:
    #===================#
    # PIVOTIZE COLUMN k #
    #===================#
    # swap in for column k the column with the largest
    # remaining column norm (same as in (weak) RRQR)
    assert v.shape[1] == n-k
    pivot( k + v.argmax() )

    while True:
      #======================================#
      # eliminate column k using householder #
      #======================================#
      elim = eliminate(k)
      if DEBUG:
        assert elim
      del elim
  
      #==========================================#
      # UPDATE inv(A) and A\B FOR NEXT ITERATION #
      #==========================================#
      update()
      if k >= n: break # <- RIGHTMOST COLUMN REACHED (NO MORE COLUMN SWAPS POSSIBLE)

      assert np.allclose(v, la.norm( R[ k:, k:], axis=0, keepdims=True))
      if DEBUG:
        assert np.allclose(Q   @ Q.T, np.eye(m))
        assert np.allclose(Q.T @ Q  , np.eye(m))
        assert np.allclose(M[:,P], Q @ R)

      assert np.allclose(0, np.tril( R[:,:k],-1))
      assert np.allclose(0, np.tril(AB[:,:k],-1))
      if DEBUG:
        assert np.allclose(sla.solve_triangular(R[:k,:k], R[:k, k:]), AB[:k, k:])
        assert np.allclose( np.eye(k),  R[:k,:k] @ AB[:k,:k] )
        assert np.allclose( np.eye(k), AB[:k,:k] @  R[:k,:k] )

      #=================================================#
      # SWAP COLUMNS THAT SIGNIFICANTLY INCREASE det(A) #
      #=================================================#

      # memorize det(A)
      logdet_a = np.sum( np.log( np.abs(np.diag(R)[:k]) ))

      # W[i,j] contains the factor by which det(A) would change
      # if column i and (j+k) of R were swapped, i.e:
      #
      #   W[i,j] = det( A(col_swap(R,i,j+k)) ) / det( A(R) )
      u = la.norm(AB[:k ,:k ], axis=1, keepdims=True)
      W = np.hypot( AB[:k,k:], u*v )
      
      i,j = np.unravel_index(W.argmax(), W.shape)
      F = W[i,j]

      if not F > f:
        if la.norm(v) <= ztol: # <- we found the correct rank
          rank,k = k,K
        break # <- handles NaN as well ... I hope

      j += k
 
      # move column i to the very right of A using cyclic permutation
      R[:,i:k-1],R[:,k-1] = R[:,i+1:k],R[:,i].copy()
      P [ i:k-1],P [ k-1] = P [ i+1:k],P [ i].copy()
      AB[ i:k-1],AB[ k-1] = AB[ i+1:k],AB[ i].copy() # <- move rows in inv(A) accordingly
 
      # re-triangulate A using Givens rotations
      for h in range(i,k-1):
        c,s = R[h:h+2,h]
        if 0 != s:
          hyp = np.hypot(c,s)
          assert 0 < hyp
          c /= hyp
          s /= hyp
          if 0 != s: # <- in case of underflow we can somewhat safely skip
            G = np.array([[ c,s],
                          [-s,c]])
            R[h:h+2,h:] = G @ R[  h:h+2,h:]
            AB[:,h:h+2] =    AB[:,h:h+2] @ G.T
            Q [:,h:h+2] =    Q [:,h:h+2] @ G.T

      R [i:,i:k] = np.triu( R[i:,i:k])
      AB[i:,i:k] = np.triu(AB[i:,i:k])

      downdate()

      # swap columns k and j
      pivot(j)
      n_swaps += 1

      # new det(A)
      logdet_A = (
          np.log( np.abs(np.diag(R)[:k]) ).sum()
        + np.log( la.norm(R[k:,k]) )
      )
      # check prediction made by W
      assert np.isclose(
        logdet_A,
        logdet_a + np.log(F)
      )

  return Q,R,P, int(rank), n_swaps
