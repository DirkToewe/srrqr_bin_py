'''
Created on Dec 22, 2019

@author: Dirk Toewe
'''

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json


def plot_results():

  with open('./srrqr_benchmark_run1.json', mode='r', encoding='utf-8') as input:
    results = json.load(input)
  del input

  fig = plt.figure()
  fig.suptitle('SRRQR Comparison: "Strong" Column Swaps against Rank')
  plot = 0

  for typ,res in results.items():
    x     = []
    y_l2r = []
    y_bin = []

    x = res['rank']
 
    def averages( y ):
      avgs = []
      prev = x[0]
      n = 0
      avg = 0
 
      for xi,yi in zip(x,y):
        if xi == prev:
          avg += yi
          n += 1
        else:
          # check that rank is sorted
          assert prev < xi
          avgs.append(avg / n)
          prev= xi
          avg = yi
          n = 1
 
      avgs.append(avg / n)
      return avgs
 
    y_l2r = averages(res['n_l2r'])
    y_bin = averages(res['n_bin'])
    x = sorted({*x})

    plot += 1
    sub = fig.add_subplot(2,2,plot)
    sub.set_title(typ, y=0)
    if 1< plot: sub.set_xlabel('rank')
    if plot in {1,3}: sub.set_ylabel('swaps')
    sub.plot(x,y_l2r, label='left to right')
    sub.plot(x,y_bin, label='binary search')

    if 2==plot:
      font = FontProperties()
      font.set_size('x-small')
      sub.legend(prop=font, loc='upper left')

  plt.show()


if __name__ == '__main__':
  plot_results()
