'''
Created on Dec 22, 2019

@author: Dirk Toewe
'''

import matplotlib.pyplot as plt
import json
import numpy as np
import resource
import time

from scipy.stats import ortho_group

# import pyximport; pyximport.install( pyimport=True, language_level=3 )
from srrqr.srrqr_l2r import srrqr_l2r
from srrqr.srrqr_bin import srrqr_bin


resource.setrlimit(resource.RLIMIT_AS, (5 * 1024**3, 6 * 1024**3))


def benchmark_swaps():
  x     = []
  y_l2r = []
  y_bin = []

  results = {}


#   def matrices():
#     print('SIZE│RANK┃N_L2R│N_BIN│ DT_L2R │ DT_BIN')
#     for n in (2**np.linspace(2,12,32)).astype(np.int):
#       for _ in range(32):
#         print('────┼────╂─────┼─────┼────────┼────────')
#         for typ,rank in (
#           ('rand',    np.random.randint(0,n+1) ),
#           ( '25%',    n*1//4                   ),
#           ( '50%',    n*2//4                   ),
#           ( '75%',    n*3//4                   ),
#           ('100%',    n*4//4                   ),
#           ('log2(n)', np.log2(n).astype(np.int))
#         ):
#           assert rank <= n
#     
#           U = ortho_group.rvs(n)
#           V = ortho_group.rvs(n)
#           S = np.random.rand(n)*1024 + 0.1
#           if rank < n:
#             if rank==0: S[rank:] = 0
#             else      : S[rank:] = np.random.rand(n-rank) * np.abs(S[:rank]).max() * np.finfo(np.float64).eps
# #           S[rank:] = 0
#     
#           A = (U*S) @ V
#           A.flags.writeable = False
#           yield A,rank,typ


  def matrices():
    n_repeat = 32

    print('SIZE│RANK┃N_L2R│N_BIN│ DT_L2R │ DT_BIN')
    for n in [32,128,512]:
#     for n in (2**np.linspace(2,12,32)).astype(np.int):

      typ = 'n=%d' % n
      print('────┼────╂─────┼─────┼────────┼────────')

      for rank in np.linspace(0,n,min(33,n+1)).astype(np.int):
        for _ in range(n_repeat):
          assert rank <= n

          U = ortho_group.rvs(n)
          V = ortho_group.rvs(n)
          S = np.random.rand(n)*1024 + 0.1
          if rank < n:
            if rank==0: S[rank:] = 0
            else      : S[rank:] = np.random.rand(n-rank) * np.abs(S[:rank]).max() * np.finfo(np.float64).eps

          A = (U*S) @ V
          A.flags.writeable = False
          yield A,int(rank),typ

      with open('./srrqr_benchmark.json', mode='w', encoding='utf-8') as out:
        json.dump(results, out, indent=2)

#       res = results[typ]
#       x = res['rank']
# 
#       def averages( y ):
#         avgs = []
#         prev = x[0]
#         avg = 0
# 
#         for xi,yi in zip(x,y):
#           if xi == prev:
#             avg += yi
#           else:
#             # check that rank is sorted
#             assert prev < xi
#             avgs.append(avg / n_repeat)
#             prev= xi
#             avg = yi
# 
#         avgs.append(avg / n_repeat)
#         return avgs
# 
#       y_l2r = averages(res['n_l2r'])
#       y_bin = averages(res['n_bin'])
#       x = sorted({*x})
# 
#       fig = plt.figure()
#       sub = fig.add_subplot(1,1,1)
#       sub.set_title(typ)
#       sub.plot(x,y_l2r, label='left to right')
#       sub.plot(x,y_bin, label='binary search')
#       sub.legend()
#       plt.show()


  for A,RANK,TYPE in matrices():
    n = A.shape[0]
    print( '{:4d}│'.format(n   ), end='' )
    print( '{:4d}┃'.format(RANK), end='' )

    def measure(srrqr):
      t0 = time.perf_counter()
      Q,R,P, rank, n_swaps  = srrqr(A)
      dt = time.perf_counter() - t0
      assert rank <= n
      assert rank == RANK, 'RANK ERROR: %d != %d' % (rank,RANK)

      assert np.allclose( np.eye(n), Q.T @ Q   )
      assert np.allclose( np.eye(n), Q   @ Q.T )
      assert np.allclose( 0, np.tril(R,-1) )
      assert np.allclose(np.sort(P), np.arange(n))
  
      assert np.allclose( A[:,P], Q @ R )
      assert np.allclose( A[:,P], Q[:,:rank] @ R[:rank] )

      return n_swaps,dt

    x.append(n)
    n_l2r,dt_l2r = measure(srrqr_l2r); y_l2r.append(n_l2r); print('{:5d}│'.format(n_l2r), end='')
    n_bin,dt_bin = measure(srrqr_bin); y_bin.append(n_bin); print('{:5d}│'.format(n_bin), end='')
    print('{:8.3f}│'.format(dt_l2r), end='')
    print('{:8.3f}' .format(dt_bin)        )

    res = results.setdefault(TYPE,{
        'size': [],
        'rank': [],
       'n_l2r': [],
       'n_bin': [],
      'dt_l2r': [],
      'dt_bin': []
    })
    res[  'size'].append(n)
    res[  'rank'].append(RANK)
    res[ 'n_l2r'].append(n_l2r)
    res[ 'n_bin'].append(n_bin)
    res['dt_l2r'].append(dt_l2r)
    res['dt_bin'].append(dt_bin)


  with open('./srrqr_benchmark.json', mode='w', encoding='utf-8') as out:
    json.dump(results, out, indent=2)

  fig = plt.figure()
  sub = fig.add_subplot(1,1,1)
  sub.set_xscale('log')
  sub.plot(x,y_l2r, label='left to right')
  sub.plot(x,y_bin, label='binary search')
  sub.legend()
  plt.show()


if __name__ == '__main__':
  benchmark_swaps()
