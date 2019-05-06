"""
1. Neighborhood-based collaborative filtering
2. Model-based collaborative filtering  

Dataset:
http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Ratings matrix is denoted by R, and it is an m × n matrix
containing m users (rows) and n movies (columns). The (i, j) 
entry of the matrix is the rating of user i for movie j and 
is denoted by r_ij
"""

import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surprise import KNNBasic, AlgoBase
from surprise.prediction_algorithms.matrix_factorization import NMF, SVD
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.model_selection import cross_validate
from surprise import Dataset

from sklearn.metrics import roc_curve

PLOT_RESULT=True

df = pd.read_csv("./ml-latest-small/ratings.csv")
movies = df['movieId'].unique()
users = df['userId'].unique()

print(f"Dataset has {movies.shape[0]} movies & {users.shape[0]} users")
movies_map = dict()
movies_inv_map = dict()
for (i, Id) in enumerate(movies):
  movies_map[Id] = i 
  movies_inv_map[i] = Id

R = np.zeros([users.shape[0], movies.shape[0]])

for idx, row in df.iterrows():
  R[int(row['userId']-1)][movies_map[row['movieId']]] =  row['rating']

print(R)

"""
Question 1: Compute the sparsity of the movie rating dataset, where sparsity is defined by:
sparsity = total num of available ratings / total num of possible ratings
"""
sparsity = len(R[R > 0]) / R.size

"""
Question 2: Plot a histogram showing the frequency of the rating values
"""
bin_width = 0.5
bin_min, bin_max = df['rating'].min(), df['rating'].max()
bins = bins = np.arange(bin_min, bin_max + bin_width, bin_width)  

if PLOT_RESULT:
  plt.figure()
  df['rating'].hist(bins=bins)
  plt.title("rating distribution")
  plt.xlabel("rating")
  plt.ylabel("number of rating")
  plt.show(0)

"""
Question 3: Plot the distribution of the number of ratings received among movies
"""
Rm = np.sum(1.0 * (R > 0), axis=0)
Rm_sorted = np.flip(np.sort(Rm))

if PLOT_RESULT:
  plt.figure()
  plt.bar(range(len(Rm_sorted)), Rm_sorted)
  plt.title("num rating distribution by movie")
  plt.xlabel("movie")
  plt.ylabel("number of rating")
  plt.grid()
  plt.show(0)

"""
Question 4: Plot the distribution of ratings among users
"""
Ru = np.sum(1.0 * (R > 0), axis=1)
Ru_sorted = np.flip(np.sort(Ru))

if PLOT_RESULT:
  plt.figure()
  plt.bar(range(len(Ru_sorted)), Ru_sorted)
  plt.title("num rating distribution by user")
  plt.xlabel("user")
  plt.ylabel("number of rating")
  plt.grid()
  plt.show(0)

"""
Question 5: Explain the salient features of the distribution found in question 3 and their 
implications for the recommendation process

Both distribution seems to be exponentially distributed 
"""

"""
Question 6: Compute the variance of the rating values received by each movie
"""
Rm_var = R.std(axis=0) ** 2
bin_min, bin_max = Rm_var.min(), Rm_var.max()
bins = bins = np.arange(bin_min, bin_max + bin_width, bin_width)  

if PLOT_RESULT:
  plt.figure()
  plt.hist(Rm_var, bins=bins)
  plt.xlabel("var of rating for each movie")
  plt.ylabel("num rating")
  plt.grid()
  plt.show(0)

"""
Question 11:
"""
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

#avg_rmse = []
#avg_mae = []
#ks = list(range(2, 100, 2))
#
#for k in ks:
#  sim_options =  { 'name': 'pearson_baseline' }
#  algo = KNNBasic(k=2, sim_options=sim_options)
#  result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5)
#  avg_rmse.append(result['test_rmse'].mean())
#  avg_mae.append(result['test_mae'].mean())

"""
Question 17
"""
#avg_rmse = []
#avg_mae = []
#
#for k in ks:
#  algo = NMF(n_factors=2)
#  result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10)
#  avg_rmse.append(result['test_rmse'].mean())
#  avg_mae.append(result['test_mae'].mean())

"""
Question 24
"""
#avg_rmse = []
#avg_mae = []
#
#for k in ks:
#  algo = SVD(n_factors=2)
#  result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10)
#  avg_rmse.append(result['test_rmse'].mean())
#  avg_mae.append(result['test_mae'].mean())

"""
Question 30: Naive Collaborative Filtering

rij_hat = mean(u_j)
"""
class NaiveCollabFilter(AlgoBase):
  def __init__(self):
    AlgoBase.__init__(self)
    self._m_uid = dict()
    
  def fit(self, trainset):
    AlgoBase.fit(self, trainset)
    self._m_uid.clear()
    for uid, iid, rating in self.trainset.all_ratings():
      if uid in self._m_uid:
        m = self._m_uid[uid][0]
        n = self._m_uid[uid][1] + 1
        m += (rating - m) / n
        self._m_uid[uid] = (m, n)
      else:
        self._m_uid[uid] = (rating, 1)
            
  def estimate(self, u, i):
    return self._m_uid[u][0] if u in self._m_uid else 0

algo = NaiveCollabFilter()
result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10)
