'''
Created on Dec 22, 2019

@author: Dirk Toewe
'''

import numpy as np
import numpy.linalg as  la
import scipy.linalg as sla


def srrqr_bin( M ):
  '''
  In contrast to `srrqr_l2r`, this algorithm does not try every potential rank
  from left to right. Instead binary search is used to find the correct rank.
  '''
  DEBUG = False

  M = np.array(M, dtype=np.float64, copy=True)
  M.flags.writeable = False

  m,n = M.shape
  f = 1.01

  assert f >= 1

  Q = np.eye(m, dtype=np.float64)
  R = M.copy()
  P = np.arange(n, dtype=np.int64)

  R_norm = la.norm(R)
  if 0 == R_norm:
    return Q,R,P, 0, 0

  # If norm(C,'fro') <= ztol, C is considered to be zero.
  # It uses max_abs(R) as scaling factor, which is always
  # less than or equal to norm(M,'fro') and it should
  # therefore be a safely low threshold.
  ztol = np.finfo(R.dtype).eps * max(m,n) * R_norm
  assert ztol >= 0

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

  AB = np.zeros_like(R) # <- keeps track of inv(A) and A\B (= inv(A) @ B)
  AB0= np.zeros_like(R)

  # k0 and K are the lower and upper bounds of the current binary search range
  k0,k,K  = 0,0,min(m,n)


  def piv_elim( j ):
    '''
    Swaps column k and j which are both right of inv(A),
    then eliminates column k using a Householder reflection.
    '''
    # SWAP
    assert k <= j
    if k < j:
      R  [:,[k,j]] = R  [:,[j,k]]
      AB [:,[k,j]] = AB [:,[j,k]]
      AB0[:,[k,j]] = AB0[:,[j,k]]
      P  [  [k,j]] = P  [  [j,k]]

    # ELIMINATE
    u = R[k:,k].copy()
    u[0] += ((u[0] > 0)*2 - 1) * la.norm(u)

    assert not 0 == u[0] # <- TODO: remove after testing

    if 0 != u[0]:
      u /= np.abs(u).max() # <- make underflow-safe (as numpy doesn't seem to guarantee that...)
      u *= np.sqrt(2) / la.norm(u)
      u  = u.reshape(-1,1)
      R[k:,:] -=            u @(u.T @ R[k:,:])
      Q[:,k:] -= (Q[:,k:] @ u)@ u.T
      R[k+1:,k] = 0


  def update( AB, k ):
    '''
    Updates inv(A) and A\B, then increments k.
    Column k needs to be eliminated for this
    method to work correctly.
    '''
    assert np.allclose(0, R[k+1:,k], rtol=0, atol=0)

    # update inv(A)
    AB[ k,  k]  = -1
    AB[:k+1,k] /= -R[k,k]

    assert k <= K
    k += 1

    # update A \ B (= inv(A) @ B)
    AB[:k,k:] += AB[:k,k-1:k] * R[k-1:k,k:]


  def downdate( AB, k ):
    '''
    Downdates inv(A) and A\B, then decrements k.
    This method reverses the changes made by `update()`.
    '''
    # downdate A \ B (= inv(A) @ B)
    AB[:k,k:] -= AB[:k,k-1:k] * R[k-1:k,k:]

    k -= 1 # <- backtrack and let loop eliminate newly swapped-in column

    # downdate inv(A)
    AB[:k+1,k ] *= -R[k,k]
    # if downgraded correctly, row k should be [-1, 0, ..., 0], see upgrade above
    assert np.allclose(-1, AB[k,k   ], rtol=0, atol=1e-3), '[k={:d}], {:f} != 1'.format( k, np.abs(AB[k,k   ]).max() )
    assert np.allclose( 0, AB[k,k+1:], rtol=0, atol=1e-3), '[k={:d}], {:f} != 0'.format( k, np.abs(AB[k,k+1:]).max() )
    AB[ k,  k:]  = 0 # <- remove cancellation errors from row k


  def adjust_k( *, increase ):
    '''
    Updates the bounds of the binary search.
    Moves k to the mid of the new search area
    to continue the binary search.
    '''
    nonlocal k0,k,K,v

    assert k0 <= k <= K

    if increase:
      # at this point we know that the rank is at least k+1, so let upade once
      assert k < K; v = la.norm(R[k:,k:], axis=0, keepdims=True)
      piv_elim( k + v.argmax() )
      update(AB,k)
      k += 1
      AB0[:k] = AB[:k]
      k0 = k
    else:
      assert k0 < k
      assert K == k # <- check that the upper bound has been adjusted
      AB[:k] = AB0[:k]
      k = k0

    mid = k0+K >> 1

    v = la.norm(R[k:,k:], axis=0, keepdims=True)

    while k < mid:
      assert v.shape[1] == n-k
      assert np.allclose(v, la.norm( R[ k:, k:], axis=0, keepdims=True))

      if la.norm(v) <= ztol:
        break

      if increase:
        piv_elim( k + v.argmax() )

      update(AB, k)
      k += 1
      v = la.norm(R[k:,k:], axis=0, keepdims=True)

  n_swaps = 0
  adjust_k(increase=True)

  while True:
    v = la.norm(R[k:,k:], axis=0, keepdims=True)

    if la.norm(v) <= ztol:
      K = k # <- remaining rows are small enough -> rank is at most k
      if k0 < k:
        # we might have gone too far already so let's go back a little
        adjust_k(increase=False)
        continue
      elif k==n:
        # we know rank=k=k0=K and no column swaps available
        break
      else:
        # we know rank=k=k0=K, but we might still find more "strong" column swaps
        assert k0 == k

    assert np.allclose(v, la.norm( R[ k:, k:], axis=0, keepdims=True))
    assert np.allclose(0, np.tril( R[:,:k],-1))
    if DEBUG:
      assert np.allclose(Q   @ Q.T, np.eye(m))
      assert np.allclose(Q.T @ Q  , np.eye(m))
      assert np.allclose(M[:,P], Q @ R)

    # CHECK AB
    assert 0 < k
    assert np.allclose(0, np.tril(AB[:,:k],-1))
    if DEBUG:
      assert np.allclose(sla.solve_triangular(R[:k,:k], R[:k, k:]), AB[:k, k:], rtol=1e-3, atol=1e-4)
      assert np.allclose( np.eye(k),  R[:k,:k] @ AB[:k,:k], rtol=0, atol=1e-3 )
      assert np.allclose( np.eye(k), AB[:k,:k] @  R[:k,:k], rtol=0, atol=1e-3 )

    # CHECK AB0
    if 0 < k0:
      assert np.allclose(0, np.tril(AB0[:,:k0],-1))
      if DEBUG:
        assert np.allclose(sla.solve_triangular(R[:k0,:k0], R[:k0, k0:]), AB0[:k0, k0:], rtol=1e-3, atol=1e-4)
        assert np.allclose( np.eye(k0),   R[:k0,:k0] @ AB0[:k0,:k0], rtol=0, atol=1e-3 )
        assert np.allclose( np.eye(k0), AB0[:k0,:k0] @   R[:k0,:k0], rtol=0, atol=1e-3 )

    #=================================================#
    # SWAP COLUMNS THAT SIGNIFICANTLY INCREASE det(A) #
    #=================================================#

    # memorize det(A)
    logdet_a = np.log2( np.abs(np.diag(R)[:k]) ).sum()

    # W[i,j] contains the factor by which det(A) would change
    # if column i and (j+k) of R were swapped, i.e:
    #
    #   W[i,j] = det( A(col_swap(R,i,j+k)) ) / det( A(R) )
    u = la.norm(AB[:k ,:k ], axis=1, keepdims=True)
    W = np.hypot( AB[:k,k:], u*v )

    # find the best "strong" column swap available
    i,j = np.unravel_index(W.argmax(), W.shape)
    F = W[i,j]

    if not F > f:
      # no more "strong" column swaps are available
      if k0 >= K:
        # nowhere else to go so rank=k=k0=K
        assert k0 == K
        assert k0 == k
        break

      # at this point remaining rows are still larger than ztol, i.e. k < rank, so let's increase k.
      adjust_k(increase=True)
      continue

    j += k

    if i < k0:
      # i INSIDE OF inv(A)_0, SO LET'S MOVE IT OUT OF THERE
      # first lets column i to column k0-1 in A
      R  [:,i:k0-1],R  [:,k0-1] = R  [:,i+1:k0],R  [:,i].copy()
      P  [  i:k0-1],P  [  k0-1] = P  [  i+1:k0],P  [  i].copy()
      AB [  i:k0-1],AB [  k0-1] = AB [  i+1:k0],AB [  i].copy() # <- move rows in inv(A)   accordingly
      AB0[  i:k0-1],AB0[  k0-1] = AB0[  i+1:k0],AB0[  i].copy() # <- move rows in inv(A)_0 accordingly

      # re-triangulate A using Givens rotations
      for h in range(i,k0-1):
        c,s = R[h:h+2,h]
        if 0 != s:
          hyp = np.hypot(c,s)
          assert 0 < hyp
          c /= hyp
          s /= hyp
          if 0 != s: # <- in case of underflow we can somewhat safely skip
            G = np.array([[ c,s],
                          [-s,c]])
            R  [  h:h+2,h:] = G @ R[  h:h+2,h:]
            AB [:,h:h+2   ] =   AB [:,h:h+2   ] @ G.T
            AB0[:,h:h+2   ] =   AB0[:,h:h+2   ] @ G.T
            Q  [:,h:h+2   ] =     Q[:,h:h+2   ] @ G.T
            R  [ h+1,h] = 0
            AB [k0-1,h] = 0
            AB0[k0-1,h] = 0

      downdate(AB0,k0)
      i = k0-1

    # move column i to the very right of A using cyclic permutation
    R  [:,i:k-1],R  [:,k-1] = R  [:,i+1:k],R  [:,i].copy()
    P  [  i:k-1],P  [  k-1] = P  [  i+1:k],P  [  i].copy()
    AB [  i:k-1],AB [  k-1] = AB [  i+1:k],AB [  i].copy() # <- move rows in inv(A) accordingly
    AB0[:,i:k-1],AB0[:,k-1] = AB0[:,i+1:k],AB0[:,i].copy() # <- move cols in (A\B)_0

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
          R [  h:h+2,h:] = G @ R[  h:h+2,h:]
          AB[:,h:h+2   ] =    AB[:,h:h+2   ] @ G.T
          Q [:,h:h+2   ] =     Q[:,h:h+2   ] @ G.T
          R  [h+1,h] = 0
          AB [k-1,h] = 0

    downdate(AB,k)
    k -= 1

    # swap columns k and j
    piv_elim(j)
    n_swaps += 1

    if i < k0:
      assert i == k0-1
      update(AB0, k0-1)

    update(AB,k)
    k += 1

    # new det(A)
    logdet_A = np.log2( np.abs(np.diag(R)[:k]) ).sum()
    # check prediction made by W
    assert np.isclose(
      logdet_A,
      logdet_a + np.log2(F),
      rtol = 1e-3,
      atol = 1e-4
    )

  return Q,R,P, int(k), n_swaps
