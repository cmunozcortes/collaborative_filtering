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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

plt.figure()
plt.hist(Rm_var, bins=bins)
plt.xlabel("var of rating for each movie")
plt.ylabel("num rating")
plt.grid()
plt.show(0)

"""
Question 7:
"""


"""
Question 8:
"""

"""
Question 9:
"""

"""
Question 10:
"""
