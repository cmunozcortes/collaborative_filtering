"""
1. Neighborhood-based collaborative filtering
2. Model-based collaborative filtering  

Dataset:
http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

"""
import pdb
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surprise import KNNBasic, AlgoBase
from surprise.prediction_algorithms.matrix_factorization import NMF, SVD
from surprise.prediction_algorithms.baseline_only import BaselineOnly
from surprise.model_selection import cross_validate, KFold, train_test_split
from surprise import Dataset, Reader, KNNWithMeans, accuracy

from sklearn.metrics import roc_curve, auc


"""
Constants
"""
PLOT_RESULT = True
USE_PICKLED_RESULTS = True 

"""
Loading data, computing rating matrix R

Ratings matrix is denoted by R, and it is an m × n matrix
containing m users (rows) and n movies (columns). The (i, j) 
entry of the matrix is the rating of user i for movie j and 
is denoted by r_ij
"""
df = pd.read_csv("./ml-latest-small/ratings.csv")
reader = Reader(rating_scale=(0.5,5))
data = Dataset.load_from_df(df[['userId','movieId','rating']], reader)

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
Question 10:
"""
# Initialize kNNWithMeans with sim options
sim_options = {
    'name': 'pearson',
    'user_based': True,
}

# Run k-NN with k=2 to k=100 in increments of 2
k_values = range(2,101,2)
results = []

if USE_PICKLED_RESULTS == True:
  with open('knn.pickle', 'rb') as handle:
    results = pickle.load(handle)
else:
  for k in k_values:
    print('\nk = {0:d}'.format(k))
    algo = KNNWithMeans(k=k, sim_options=sim_options)
    results.append(cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=10, 
                                verbose=True, n_jobs=-1))
  # Pickle results
  with open('knn.pickle', 'wb') as handle:
    pickle.dump(results, handle)

# Calculate mean scores
mean_scores = np.zeros((50,2))
for counter, result in enumerate(results):
  mean_scores[counter,0] = np.mean(result['test_rmse'])
  mean_scores[counter,1] = np.mean(result['test_mae'])

# Print steady-state value for RMSE and MAE
print('\nRMSE steady-state value: {:.3f}'.format(mean_scores[20,0]))
print('MAE steady-state value: {:.3f}'.format(mean_scores[20,1]))

# Plot mean scores
if PLOT_RESULT:
  # Plot RMSE
  plt.figure(figsize=(15,5))
  plt.subplot(1,2,1)
  plt.plot(k_values, mean_scores[:,0],'-x')
  plt.title('Mean RMSE for k-NN with Cross Validation')
  plt.ylabel('Mean RSME')
  plt.xlabel('Number of $k$ neighbors')

  # Plot MAE
  plt.subplot(1,2,2)
  plt.plot(k_values, mean_scores[:,1],'-x')
  plt.title('Mean MAE for k-NN with Cross Validation')
  plt.ylabel('Mean MAE')
  plt.xlabel('Number of $k$ neighbors')
  plt.tight_layout()
  plt.show()

"""
Question 12: k-NN on popular movies
"""
# Create a dict where each movieId is a key and the values are a list
# of all the ratings for the movieId
ratings = {}
for row in data.raw_ratings:
  # if movieId not in dict, add it
  if row[1] not in ratings:
    ratings[row[1]] = []
    
  # Add ratings to movieId list
  ratings[row[1]].append(row[2])

# Create dictionary with rating variance for each movieId
variances = {}
for movieId in ratings:
  variances[movieId] = np.var(ratings[movieId])

# Create list with movies with more than 2 ratings
pop_movies = [movie for movie in ratings if len(ratings[movie]) > 2]

# Train/test using cross-validation iterators
kf = KFold(n_splits=10)
k_rmse = 0
rmse_pop = []

if USE_PICKLED_RESULTS == True:
  with open('knn_pop.pickle', 'rb') as handle:
    rmse_pop = pickle.load(handle)
else:
  # Iterate over all k values and calculate RMSE for each
  for k in k_values:
    algo = KNNWithMeans(k=k, sim_options=sim_options)
    for counter, [trainset, testset] in enumerate(kf.split(data)):
      print('\nk = {0:d}, fold = {1:d}'.format(k, counter+1))
      
      # Train algorithm with 9 unmodified trainsets
      algo.fit(trainset)
    
      # Test with trimmed test set
      trimmed_testset = [x for x in testset if x[1] in pop_movies]
      predictions = algo.test(trimmed_testset)
    
      # Compute and print Root Mean Squared Error (RMSE) for each fold
      k_rmse += accuracy.rmse(predictions, verbose=True)
  
    #Compute mean of all rsme values for each k
    print('Mean RMSE for 10 folds: ', k_rmse/(counter+1))
    rmse_pop.append(k_rmse / (counter+1))
    k_rmse = 0
  
  # Pickle results
  with open('knn_pop.pickle', 'wb') as handle:
    pickle.dump(rmse_pop, handle)

# Print minimum RMSE
print('\nPopular Movies:')
print('Minimum average RMSE: {:.3f}'.format(np.min(rmse_pop)))


if PLOT_RESULT:
  # Plot RMSE versus k
  plt.plot(k_values, rmse_pop, '-x')
  plt.title('Average RMSE over $k$ with 10-fold cross validation')
  plt.xlabel('$k$ Nearest Neighbors')
  plt.ylabel('Average RMSE')
  plt.show()

"""
Question 13: Unpopular movie trimmed set
"""
rmse_unpop = []
if USE_PICKLED_RESULTS == True:
  with open('knn_unpop.pickle', 'rb') as handle:
    rmse_unpop = pickle.load(handle)
else: 
  for k in k_values:
    algo = KNNWithMeans(k=k, sim_options=sim_options)
    for counter, [trainset, testset] in enumerate(kf.split(data)):
      print('\nk = {0:d}, fold = {1:d}'.format(k, counter+1))
    
      # Train algorithm with 9 unmodified trainset
      algo.fit(trainset)
    
      # Test with trimmed test set
      trimmed_testset = [x for x in testset if x[1] not in pop_movies]
      predictions = algo.test(trimmed_testset)
    
      # Compute and print Root Mean Squared Error (RMSE) for each fold
      k_rmse += accuracy.rmse(predictions, verbose=True)
  
    #Compute mean of all rsme values for each k
    print('Mean RMSE for 10 folds: ', k_rmse/(counter+1))
    rmse_unpop.append(k_rmse / (counter+1))
    k_rmse = 0

  # Pickle results
  with open('knn_unpop.pickle', 'wb') as handle:
    pickle.dump(rmse_unpop, handle)

# Print minimum RMSE
print('\nUnpopular Movies:')
print('Minimum average RMSE: {:.3f}'.format(np.min(rmse_unpop)))

if PLOT_RESULT:
  # Plot RMSE versus k
  plt.plot(k_values, rmse_unpop, '-x')
  plt.title('Average RMSE over $k$ with 10-fold cross validation')
  plt.xlabel('$k$ Nearest Neighbors')
  plt.ylabel('Average RMSE')
  plt.show()

"""
Question 14: Trimmed test set - movies with more than 5 ratings and variance higher
than 2.
"""
# Create list with high_variance movies
high_var_movies = [movieId for movieId in ratings if len(ratings[movieId]) >=5
                   and variances[movieId] >= 2]

# Empty list to store rmse for each k
rmse_high_var = []

# Using cross-validation iterators
kf = KFold(n_splits=10)
k_rmse = 0

if USE_PICKLED_RESULTS == True:
  with open('knn_var.pickle', 'rb') as handle:
    rmse_high_var = pickle.load(handle)
else:
  for k in k_values:
    algo = KNNWithMeans(k=k, sim_options=sim_options)
    for counter, [trainset, testset] in enumerate(kf.split(data)):
      print('\nk = {0:d}, fold = {1:d}'.format(k, counter+1))
    
      # Train algorithm with 9 unmodified trainset
      algo.fit(trainset)
    
      # Test with trimmed test set
      trimmed_testset = [x for x in testset if x[1] in high_var_movies]
      predictions = algo.test(trimmed_testset)
    
      # Compute and print Root Mean Squared Error (RMSE) for each fold
      k_rmse += accuracy.rmse(predictions, verbose=True)
  
    # Compute mean of all rsme values for each k
    print('Mean RMSE for 10 folds: ', k_rmse/(counter+1))
    rmse_high_var.append(k_rmse / (counter+1))
    k_rmse = 0
  
  # Pickle results
  with open('knn_var.pickle', 'wb') as handle:
    pickle.dump(rmse_high_var, handle)

# Print minimum RMSE
print('\nHigh-Variance Movies:')
print('Minimum average RMSE: {:.3f}\n'.format(np.min(rmse_high_var)))

if PLOT_RESULT:
  # Plot RMSE versus k
  plt.plot(k_values, rmse_high_var, '-x')
  plt.title('Average RMSE over $k$ with 10-fold cross validation')
  plt.xlabel('$k$ Nearest Neighbors')
  plt.ylabel('Average RMSE')
  plt.show()

"""
Question 15:
"""
k = 20  # best k value found in question 10
threshold_values = [2.5, 3, 3.5, 4]
roc_results = []

for threshold in threshold_values:
  train_set, test_set = train_test_split(data, test_size = 0.1)
  algo = KNNWithMeans(k=k, sim_options=sim_options)
  algo.fit(train_set)
  predictions = algo.test(test_set)
  
  # r_ui is the 'true' rating
  y_true = [0 if prediction.r_ui < threshold else 1
                 for prediction in predictions]
  # 'est' is the estimated rating
  y_score = [prediction.est for prediction in predictions]
  fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
  roc_auc = auc(fpr, tpr)
  roc_results.append((fpr, tpr, roc_auc, threshold))

# Plot ROC and include area under curve
if PLOT_RESULT == True:
  plt.figure(figsize=(15,10))
  lw = 2
  for i, result in enumerate(roc_results):
    plt.subplot(2,2,i+1)
    plt.plot(result[0], result[1], color='darkorange', lw=lw, 
             label='ROC curve (area = %0.2f)' % result[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Threshold = {:.1f}'.format(result[3]), fontsize='xx-large')
    plt.legend(loc="lower right", fontsize='xx-large')
  plt.tight_layout()
  plt.show()



"""
Question 17
"""
kf = KFold(n_splits=10)
rmse, mae = 0, 0
kf_rmse, kf_mae = [], []

k_values = range(2,101,2)
for k in k_values:
  algo = NMF(n_factors=k)
  for trainset, testset in kf.split(data):
    algo.fit(trainset)
    pred = algo.test(testset)
    rmse += accuracy.rmse(pred)
    mae += accuracy.mae(pred)
  kf_rmse.append(rmse / kf.n_splits)
  kf_rmse.append(mae / kf.n_splits)

if True:
  plt.figure()
  plt.plot(k_values, kf_rmse, '-x')
  plt.title('Average RMSE over $k$ with 10-fold cross validation')
  plt.xlabel('n_factors')
  plt.ylabel('Average RMSE')

  plt.figure()
  plt.plot(k_values, kf_mae, '-x')
  plt.title('Average MAE over $k$ with 10-fold cross validation')
  plt.xlabel('n_factors')
  plt.ylabel('Average MAE')

"""
Question 18
"""

movieDat = pd.read_csv("./ml-latest-small/movies.csv")

indivGenre = []

# write individual genres into a new array
for g in movieDat['genres']:
    for i in g.split('|'):
        indivGenre.append(i)

# list unique individual genres
np.unique(indivGenre)

"""
Question 19: NNMF on Popular Movies
"""
# Using cross-validation iterators
kf = KFold(n_splits=10)
k_rmse = 0
rmse_pop = []

# Iterate over all k values and calculate RMSE for each
for k in k_values:
  algo = NMF(n_factors=k)
  for counter, [trainset, testset] in enumerate(kf.split(data)):
    print('\nk = {0:d}, fold = {1:d}'.format(k, counter+1))
    
    # Train algorithm with 9 unmodified trainsets
    algo.fit(trainset)
    
    # Test with trimmed test set
    trimmed_testset = [x for x in testset if x[1] in pop_movies]
    predictions = algo.test(trimmed_testset)
    
    # Compute and print Root Mean Squared Error (RMSE) for each fold
    k_rmse += accuracy.rmse(predictions, verbose=True)
  
  #Compute mean of all rmse values for each k
  print('Mean RMSE for 10 folds: ', k_rmse/(counter+1))
  rmse_pop.append(k_rmse / (counter+1))
  k_rmse = 0

print('RMSE values:')
print(rmse_pop)

# Plot RMSE versus k
plt.plot(k_values, rmse_pop, '-x')
plt.title('Average RMSE over $k$ with 10-fold cross validation')
plt.xlabel('$k$ Nearest Neighbors')
plt.ylabel('Average RMSE')

"""
Question 20: NNMF on Unpopular Movies
"""
# Train/test using cross-validation iterators
kf = KFold(n_splits=10)
k_rmse = 0

rmse_unpop = []

for k in k_values:
    algo = NMF(n_factors=k, biased=False)
    
    for counter, [trainset, testset] in enumerate(kf.split(data)):
      print('\nk = {0:d}, fold = {1:d}'.format(k, counter+1))
    
      # Train algorithm with 9 unmodified trainset
      algo.fit(trainset)
    
      # Test with unpopular movie trimmed test set
      trimmed_testset = [x for x in testset if x[1] not in pop_movies]
      predictions = algo.test(trimmed_testset)
    
      # Compute and print Root Mean Squared Error (RMSE) for each fold
      k_rmse += accuracy.rmse(predictions, verbose=True)
  
    #Compute mean of all rsme values for each k
    print('Mean RMSE for 10 folds: ', k_rmse/(counter+1))
    rmse_unpop.append(k_rmse / (counter+1))
    k_rmse = 0  


# Plot RMSE versus k
plt.plot(k_values, rmse_unpop, '-x')
plt.title('Unpopular Test Set: Average RMSE over $k$ with 10-fold cross validation')
plt.xlabel('$k$ Nearest Neighbors')
plt.ylabel('Average RMSE')

"""
Question 21: NNMF on High Variance Movies
"""
# Empty list to store rmse for each k
rmse_high_var = []

# Using cross-validation iterators
kf = KFold(n_splits=10)
k_rmse = 0

for k in k_values:
    algo = NMF(n_factors=k, biased=False)
    for counter, [trainset, testset] in enumerate(kf.split(data)):
      print('\nk = {0:d}, fold = {1:d}'.format(k, counter+1))
    
      # Train algorithm with 9 unmodified trainset
      algo.fit(trainset)
    
      # Test with trimmed test set
      trimmed_testset = [x for x in testset if x[1] in high_var_movies]
      predictions = algo.test(trimmed_testset)
    
      # Compute and print Root Mean Squared Error (RMSE) for each fold
      k_rmse += accuracy.rmse(predictions, verbose=True)
  
    # Compute mean of all rsme values for each k
    print('Mean RMSE for 10 folds: ', k_rmse/(counter+1))
    rmse_high_var.append(k_rmse / (counter+1))
    k_rmse = 0  


# Plot RMSE versus k
plt.plot(k_values, rmse_high_var, '-x')
plt.title('High Variance: Average RMSE over $k$ with 10-fold cross validation')
plt.xlabel('$k$ Nearest Neighbors')
plt.ylabel('Average RMSE')

"""
Question 22: NNMF ROC Plots
"""
def plotROC(fpr, tpr, roc_auc, threshold):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: Threshold = %s' %threshold)
    plt.legend(loc="lower right")
    plt.show()

k = 20  # best k value found in question 18
threshold_values = [2.5, 3, 3.5, 4]

for threshold in threshold_values:
  train_set, test_set = train_test_split(data, test_size = 0.1)
  algo = NMF(n_factors=k, biased=False)
  algo.fit(train_set)
  predictions = algo.test(test_set)
  
  # r_ui is the 'true' rating
  y_true = [0 if prediction.r_ui < threshold else 1
                 for prediction in predictions]
  # est is the estimated rating
  y_score = [prediction.est for prediction in predictions]
  fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
  roc_auc = auc(fpr, tpr)

  plotROC(fpr, tpr, roc_auc, threshold)

"""
Question 23: Movie-Latent Factor Interaction
"""
reader = Reader(rating_scale=(0.5,5))
data = Dataset.load_from_df(df[['userId','movieId','rating']], reader)
data = data.build_full_trainset()

movieDat = pd.read_csv('ml-latest-small/movies.csv')

nmf = NMF(n_factors=20, biased=False)
nmf.fit(data)

movies = df['movieId'].unique()  # identify unique movie IDs from the ratings CSV (9724, already sorted)
V = nmf.qi

# get top 10 movie genres for the first 20 columns of the V matrix
for i in range(20):
    Vcol = V[:,i]
    
    # convert column of V into a list for processing
    VcolOrig = []
    VcolSort = []
    for j in range(len(Vcol)):
        VcolOrig.append(Vcol[j]) # original array for looking up movie index
        VcolSort.append(Vcol[j]) # sorted array for getting top movies
    
    # sort Vcolumn list in descending order
    VcolSort.sort(reverse=True)
    
    print('\nIn the %i column, the top 10 movie genres are:' %(i+1))
    
    for k in range(10):
        movIndex = VcolOrig.index(VcolSort[k])
        movID = movies[movIndex]
        genre = movieDat.loc[movieDat['movieId']==movID]['genres'].values
        print(' %i) ' %(k+1), genre)

"""
Question 24
"""
rmse, mae = 0, 0
kf_rmse, kf_mae = [], []

k_values = range(2,101,2)
for k in k_values:
  algo = SVD(n_factors=k)
  for trainset, testset in kf.split(data):
    algo.fit(trainset)
    pred = algo.test(testset)
    rmse += accuracy.rmse(pred)
    mae += accuracy.mae(pred)
  kf_rmse.append(rmse / kf.n_splits)
  kf_rmse.append(mae / kf.n_splits)

if True:
  plt.figure()
  plt.plot(k_values, kf_rmse, '-x')
  plt.title('Average RMSE over $k$ with 10-fold cross validation')
  plt.xlabel('n_factors')
  plt.ylabel('Average RMSE')

  plt.figure()
  plt.plot(k_values, kf_mae, '-x')
  plt.title('Average MAE over $k$ with 10-fold cross validation')
  plt.xlabel('n_factors')
  plt.ylabel('Average MAE')

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
algo.fit(data.build_full_trainset())

kf = KFold(n_splits=10)
kf_rmse = []
for _, testset in kf.split(data):
  pred = algo.test(testset)
  kf_rmse.append(accuracy.rmse(pred, verbose=True))
print('Naive Collab Fillter RMSE for 10 folds CV: ', np.mean(kf_rmse))

"""
Question 31: 
"""
kf_rmse = []
for _, testset in kf.split(data):
  trimmed_testset = [x for x in testset if x[1] in pop_movies]
  pred = algo.test(trimmed_testset)
  kf_rmse.append(accuracy.rmse(pred, verbose=True))
print('Naive Collab Fillter RMSE for 10 folds CV (popular testset): ', np.mean(kf_rmse))

"""
Question 32: 
"""
kf_rmse = []
for _, testset in kf.split(data):
  trimmed_testset = [x for x in testset if x[1] not in pop_movies]
  pred = algo.test(trimmed_testset)
  kf_rmse.append(accuracy.rmse(pred, verbose=True))
print('Naive Collab Fillter RMSE for 10 folds CV (not popular testset): ', np.mean(kf_rmse))

"""
Question 33: 
"""
kf_rmse = []
for _, testset in kf.split(data):
  trimmed_testset = [x for x in testset if x[1] in high_var_movies]
  pred = algo.test(trimmed_testset)
  kf_rmse.append(accuracy.rmse(pred, verbose=True))
print('Naive Collab Fillter RMSE for 10 folds CV (high var testset): ', np.mean(kf_rmse))

"""
Question 34
Plot the ROC curves (threshold = 3) for the k-NN, NNMF, and
MF with bias based collaborative filters in the same figure. Use the figure to
compare the performance of the filters in predicting the ratings of the movies.

k-NN : k = 20
NNMF : k = 18 or 20
MF   : k = 20 
"""
trainset, testset = train_test_split(data, test_size = 0.1)
sim_options = {
  'name': 'pearson',
  'user_based': True,
}
knn = KNNWithMeans(k=20, sim_options=sim_options)
nmf = NMF(n_factors=20, biased=False)
svd = SVD(n_factors=20)

plt.figure()
threshold = 3
algos = (('kNN', knn), ('NMMF', nmf), ('MF', svd))
for name, algo in algos:
  algo.fit(trainset)
  pred = algo.test(testset)
  y_true  = [0 if p.r_ui < threshold else 1 for p in pred]
  y_score = [p.est for p in pred]
  fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
  roc_auc = auc(fpr, tpr)
  label =  name + ' ROC curve (area = %0.2f)' % roc_auc
  plt.plot(fpr, tpr, label=label)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Threshold = {:.1f}'.format(threshold))
plt.legend()
plt.show(0)

"""
Question 36: Evaluating ranking using precision-recall curve

Plot average precision (Y-axis) against t (X-axis) for the rank-
ing obtained using k-NN collaborative lter predictions. Also, plot the average
15 recall (Y-axis) against t (X-axis) and average precision (Y-axis) against average
recall (X-axis). Use the k found in question 11 and sweep t from 1 to 25 in step
sizes of 1. For each plot, briefly comment on the shape of the plot.
"""
# TODO: implement this function according to section 8.1 and 8.2 in project 
# handout:
# 1. Sort prediction list in descending order
# 2. Select the first t-items from the sorted list to recommend to the user
# 3. In the set of 't' recommended items to the user, drop the items for which we 
#    don't have a ground truth rating
# 4. Calculate precision and recall according to eqns (12) and (13)
def evaluate_ranking(predictions):
  # Empty dictionaries to store precision and recall for each userId
  precision = {}
  recall = {}
  return precision, recall

# Recommended movies set sizes
t_size = range(1,26)

# Empty lists to store precision and recall for each 't'
precision = []
recall = []

# For each t size follow pseudo-code provided in 
# section 8.1 of project handout
for t in t_size:
  for trainset, testset in kf.split(data):
    # Using knn algo defined above in Q34
    knn.fit(trainset)
    pred = knn.test(testset)

    # Rank predictions and evaluate
    kf_precision, kf_recall = evaluate_ranking(pred)
    precision.append(kf_precision)
    recall.append(kf_recall)

"""
Question 37
"""

"""
Question 38
"""