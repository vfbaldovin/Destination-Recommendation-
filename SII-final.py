import pandas as pd
import numpy as np
import os
import warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from surprise import SVD, NormalPredictor, KNNBasic, KNNWithMeans, KNNWithZScore, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
warnings.filterwarnings("ignore", category = RuntimeWarning)

os.getcwd()
os.chdir("C:\\Users\\40743\\Desktop\\Facultate\\Proiect")

input_data = pd.read_csv(r'google_review_ratings.csv')

input_data.shape

input_data.columns

input_data.head(5)

input_data.info()

input_data.drop('Unnamed: 25', axis = 1, inplace = True)

column_names = ['user_id', 'churches', 'resorts', 'beaches', 'parks', 'theatres', 'museums', 'malls', 'zoo', 'restaurants', 'pubs_bars', 'local_services', 'burger_pizza_shops', 'hotels_other_lodgings', 'juice_bars', 'art_galleries', 'dance_clubs', 'swimming_pools', 'gyms', 'bakeries', 'beauty_spas', 'cafes', 'view_points', 'monuments', 'gardens']
input_data.columns = column_names

input_data[column_names].isnull().sum()

input_data = input_data.fillna(0)

input_data.dtypes

input_data['local_services'][input_data['local_services'] == '2\t2.']

local_services_mean = input_data['local_services'][input_data['local_services'] != '2\t2.']
input_data['local_services'][input_data['local_services'] == '2\t2.'] = np.mean(local_services_mean.astype('float64'))
input_data['local_services'] = input_data['local_services'].astype('float64')

input_data.dtypes

input_data[column_names[:12]].describe()
print(input_data[column_names[:12]].describe())
input_data[column_names[12:]].describe()
print(input_data[column_names[12:]].describe())

input_data_description = input_data.describe()
min_val = input_data_description.loc['min'] > 0
min_val[min_val]


import matplotlib.pyplot as plt
import numpy as np
plt.rcdefaults()
# %matplotlib inline
no_of_zeros = input_data[column_names[1:]].astype(bool).sum(axis=0).sort_values()

plt.figure(figsize=(10,7))
plt.barh(np.arange(len(column_names[1:])), no_of_zeros.values, align='center', alpha=0.5)
plt.yticks(np.arange(len(column_names[1:])), no_of_zeros.index)
plt.xlabel('No of reviews')
plt.ylabel('Categories')
plt.title('No of reviews under each category')
plt.show()


no_of_reviews = input_data[column_names[1:]].astype(bool).sum(axis=1).value_counts()

plt.figure(figsize=(10,7))
plt.bar(np.arange(len(no_of_reviews)), no_of_reviews.values, align='center', alpha=0.5)
plt.xticks(np.arange(len(no_of_reviews)), no_of_reviews.index)
plt.ylabel('No of reviews')
plt.xlabel('No of categories')
plt.title('No of Categories vs No of reviews')

plt.show()

avg_rating = input_data[column_names[1:]].mean()
avg_rating = avg_rating.sort_values()

plt.figure(figsize=(10,7))
plt.barh(np.arange(len(column_names[1:])), avg_rating.values, align='center', alpha=0.5)
plt.yticks(np.arange(len(column_names[1:])), avg_rating.index)
plt.xlabel('Average Rating')
plt.title('Average rating per Category')
plt.show()


entertainment = ['theatres', 'dance_clubs', 'malls']
food_travel = ['restaurants', 'pubs_bars', 'burger_pizza_shops', 'juice_bars', 'bakeries', 'cafes']
places_of_stay = ['hotels_other_lodgings', 'resorts']
historical = ['churches', 'museums', 'art_galleries', 'monuments']
nature = ['beaches', 'parks', 'zoo', 'view_points', 'gardens']
services = ['local_services', 'swimming_pools', 'gyms', 'beauty_spas']


df_category_reviews = pd.DataFrame(columns = ['entertainment', 'food_travel', 'places_of_stay', 'historical', 'nature', 'services'])


df_category_reviews['entertainment'] = input_data[entertainment].mean(axis = 1)
df_category_reviews['food_travel'] = input_data[food_travel].mean(axis = 1)
df_category_reviews['places_of_stay'] = input_data[places_of_stay].mean(axis = 1)
df_category_reviews['historical'] = input_data[historical].mean(axis = 1)
df_category_reviews['nature'] = input_data[nature].mean(axis = 1)
df_category_reviews['services'] = input_data[services].mean(axis = 1)


df_category_reviews.describe()

'''
Approach 1: Popularity Based Recommendation Engine

'''

# ratings_per_category_df = pd.DataFrame(input_data[column_names[1:]].mean()).reset_index(level=0)

# ratings_per_category_df.columns = ['category', 'avg_rating']

# ratings_per_category_df['no_of_ratings'] = input_data[column_names[1:]].astype(bool).sum(axis=0).values.tolist()

# scaler = MinMaxScaler()
# ratings_per_category_df['avg_rating_scaled'] = scaler.fit_transform(ratings_per_category_df['avg_rating'].values.reshape(-1,1))
# ratings_per_category_df['no_of_ratings_scaled'] = scaler.fit_transform(ratings_per_category_df['no_of_ratings'].values.reshape(-1,1))

# def calculate_weighted_rating(x):
#     return (0.5 * x['avg_rating_scaled'] + 0.5 * x['no_of_ratings_scaled'])

# ratings_per_category_df['weighted_rating'] = ratings_per_category_df.apply(calculate_weighted_rating, axis = 1)
# ratings_per_category_df = ratings_per_category_df.sort_values(by=['weighted_rating'], ascending = False)

# input_data.head()

# def get_recommendation_based_on_popularity(x):
#     zero_cols = input_data[input_data['user_id'] == x['user_id']][column_names[1:]].astype(bool).sum(axis=0)
#     zero_df = pd.DataFrame(zero_cols[zero_cols == 0]).reset_index(level = 0)
#     zero_df.columns = ['category', 'rating']
#     zero_df = pd.merge(zero_df, ratings_per_category_df, on = 'category', how = 'left')[['category', 'weighted_rating']]
#     zero_df = zero_df.sort_values(by = ['weighted_rating'], ascending = False)
#     if len(zero_df) > 0:
#         return zero_df['category'].values[0]
#     else:
#         return ""

# input_data_recommendation = input_data.copy()
# input_data_recommendation['recommendation_based_on_popularity'] = input_data_recommendation.apply(get_recommendation_based_on_popularity, axis = 1)

# input_data_recommendation['recommendation_based_on_popularity'][input_data['user_id'] == "User 16"]
# print(input_data_recommendation['recommendation_based_on_popularity'][input_data['user_id'] == "User 16"])

'''
Approach 2: Recommender based on kNN
'''

input_data_matrix = input_data[column_names[1:]].values
knn_model = NearestNeighbors(n_neighbors=5).fit(input_data_matrix)

query_index = np.random.choice(input_data[column_names[1:]].shape[0])
distances, indices = knn_model.kneighbors(input_data[column_names[1:]].iloc[query_index, :].values.reshape(1,-1), n_neighbors = 5)

def compare(index, ind):        
    zero_cols_in = input_data.loc[index].astype(bool)
    zero_df_in = pd.DataFrame(zero_cols_in[zero_cols_in == True]).reset_index(level = 0)
    in_wo_rating = zero_df_in['index']
    sug_user = input_data.loc[ind]
    zero_cols_sug = sug_user.astype(bool)
    zero_df_sug = pd.DataFrame(zero_cols_sug[zero_cols_sug == True]).reset_index(level = 0)
    sug_wo_rating = zero_df_sug['index']
    sugg_list = list(set(sug_wo_rating) - set(in_wo_rating))
    return sugg_list
def recommend_knn(index):
    distances, indices = knn_model.kneighbors(input_data[column_names[1:]].iloc[index, :].values.reshape(1,-1), n_neighbors = 10)
    distances = np.sort(distances)
    for i in range(0,len(indices[0])):
        ind = np.where(distances.flatten() == distances[0][i])[0][0]
        sug_list = compare(index, indices[0][i]) 
        if len(sug_list) > 0:
            break
    return sug_list
print(recommend_knn(16))          

'''
Approach 3: Recommender Based on Matrix Factorization

'''
input_data_matrix = input_data.set_index('user_id').values
user_ratings_mean = np.mean(input_data_matrix, axis = 1)
user_ratings_demeaned = input_data_matrix - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(user_ratings_demeaned, k = 1)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = column_names[1:])
preds_df.head()

def recommend_svd(index):
    zero_cols_in = input_data.loc[index].astype(bool)
    zero_df_in = pd.DataFrame(zero_cols_in[zero_cols_in == False]).reset_index(level = 0)
    in_wo_rating = zero_df_in['index']
    sug_user = preds_df[in_wo_rating.values.tolist()[1:]].loc[index]
    sug_list = sug_user.sort_values(ascending = False).index[0]
    return sug_list
print(recommend_svd(16))


'''
Approach 4: Clustering the data for user segmentation

'''


scaler = MinMaxScaler()
# print(input_data)
ratings_per_category_df = pd.DataFrame(input_data[column_names[1:]].mean()).reset_index(level=0)

# ratings_per_category_df.columns = ['category', 'avg_rating']

ratings_per_category_df['no_of_ratings'] = input_data[column_names[1:]].astype(bool).sum(axis=0).values.tolist()
print('sad', input_data)
print('bbb', input_data[column_names[1:]].values)

input_array = scaler.fit_transform(input_data[column_names[1:]].values)
ratings_per_category_df['no_of_ratings_scaled'] = scaler.fit_transform(ratings_per_category_df['no_of_ratings'].values.reshape(-1,1))
#nput_array = input_data[column_names[1:]].values
print('HALELELE', len(input_array), len(input_array[0]))
kmeans = KMeans(n_clusters=6)
# fit kmeans object to data
kmeans.fit(input_array)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.predict(input_array)

plt.scatter(input_array[y_km ==0,0], input_array[y_km == 0,1], s=100, c='red')
plt.scatter(input_array[y_km ==1,0], input_array[y_km == 1,1], s=100, c='black')
plt.scatter(input_array[y_km ==2,0], input_array[y_km == 2,1], s=100, c='blue')
plt.scatter(input_array[y_km ==3,0], input_array[y_km == 3,1], s=100, c='cyan')
plt.show()

Sum_of_squared_distances = []
K = range(1,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(input_array)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


for n_clusters in range(2,30):
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(input_array)
    centers = clusterer.cluster_centers_

    score = silhouette_score (input_array, preds)
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))


'''
Approach 5: Recommender based on Surprise python package to calculate evaluation metrics

'''

reader = Reader(rating_scale=(0, 5))
df = input_data.replace(0, np.nan).set_index('user_id', append=True).stack().reset_index().rename(columns={0:'rating', 'level_2':'itemID', 'user_id':'userID'}).drop('level_0',1)
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)


benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), NormalPredictor(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), CoClustering()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
bench_mark_df = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')