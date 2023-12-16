# import the required libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv("rating.csv")
ratings = ratings.drop("timestamp", axis=1)

ratings["userId"] = ratings["userId"].fillna(0)
ratings["movieId"] = ratings["movieId"].fillna(0)

ratings["rating"] = ratings["rating"].fillna(ratings["rating"].mean())

frac = 0.0001  # change this to use relatively smaller amount of data accordingly.

small_data = ratings.sample(frac=frac, random_state=42)
print(small_data.info())

train_data, test_data = train_test_split(small_data, test_size=0.2)

train_data_matrix = train_data[["userId", "movieId", "rating"]].to_numpy()
test_data_matrix = test_data[["userId", "movieId", "rating"]].to_numpy()


print(train_data_matrix.shape)
print(test_data_matrix.shape)

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(train_data, metric="correlation")
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation[:4, :4])

# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric="correlation")
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation[:4, :4])


def predict(ratings, similarity, type="user"):
    if type == "user":
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = ratings - mean_user_rating[:, np.newaxis]
        pred = (
            mean_user_rating[:, np.newaxis]
            + similarity.dot(ratings_diff)
            / np.array([np.abs(similarity).sum(axis=1)]).T
        )
    elif type == "item":
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# Function to calculate RMSE
def rmse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


# Predict ratings on the training data with both similarity score
user_prediction = predict(train_data_matrix, user_correlation, type="user")
item_prediction = predict(train_data_matrix, item_correlation, type="item")

# RMSE on the test data
print("User-based CF RMSE: " + str(rmse(user_prediction, test_data_matrix)))
print("Item-based CF RMSE: " + str(rmse(item_prediction, test_data_matrix)))
# RMSE on the train data
print("User-based CF RMSE: " + str(rmse(user_prediction, train_data_matrix)))
print("Item-based CF RMSE: " + str(rmse(item_prediction, train_data_matrix)))

# Using cosine similarity
print("Collaborative Filtering Using Cosine Similarity")


def eval_item(userid, number):
    # Assuming user_predicted_ratings is a NumPy array
    user_id = userid
    sorted_indices = np.argsort(item_predicted_ratings[user_id])[::-1]
    top_movie_indices = sorted_indices[:number]
    top_movies = ratings["movieId"].iloc[top_movie_indices]
    return top_movies


def eval_user(userid, number):
    # Assuming user_predicted_ratings is a NumPy array
    user_id = userid
    sorted_indices = np.argsort(user_predicted_ratings[user_id])[::-1]
    top_movie_indices = sorted_indices[:number]
    top_movies = ratings["movieId"].iloc[top_movie_indices]
    return top_movies


# User Similarity Matrix using Cosine similarity as a similarity measure between Users
def cosine_user(data):
    item_similarity = cosine_similarity(data)
    item_similarity[np.isnan(item_similarity)] = 0
    user_predicted_ratings = np.dot(user_similarity, data)
    return (
        user_similarity,
        user_similarity.shape,
        user_predicted_ratings,
        user_predicted_ratings.shape,
    )


# Item Similarity Matrix using Cosine similarity as a similarity measure between Items
def cosine_item(data):
    item_similarity = cosine_similarity(data)
    item_similarity[np.isnan(item_similarity)] = 0
    item_predicted_ratings = np.dot(data.T, item_similarity)
    return (
        item_similarity,
        item_similarity.shape,
        item_predicted_ratings,
        item_predicted_ratings.shape,
    )


movie_features = train_data.pivot(
    index="movieId", columns="userId", values="rating"
).fillna(0)
user_data = train_data.pivot(index="userId", columns="movieId", values="rating").fillna(
    0
)

# Item Similarity Matrix using cosine_item function
(
    item_similarity,
    item_similarity.shape,
    item_predicted_ratings,
    item_predicted_ratings.shape,
) = cosine_item(movie_features)

# evaluate using eval_item function
top_movies_predicted = eval_item(23, 5)

# User Similarity Matrix using cosine_user function
(
    user_similarity,
    user_similarity.shape,
    user_predicted_ratings,
    user_predicted_ratings.shape,
) = cosine_user(user_data)

# Evaluate using eval_item function
top_movies_predicted = eval_user(23, 5)
