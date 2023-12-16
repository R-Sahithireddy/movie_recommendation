import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam

# Load MovieLens dataset
ratings = pd.read_csv("rating.csv")
ratings = ratings.drop("timestamp", axis=1)

# Label encode userId and movieId
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings["userId"] = user_encoder.fit_transform(ratings["userId"])
ratings["movieId"] = movie_encoder.fit_transform(ratings["movieId"])

# Train-test split
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Number of unique users and movies
n_users = ratings["userId"].nunique()
n_movies = ratings["movieId"].nunique()

# Embedding size
embedding_size = 50

# Define the model
user_input = Input(shape=(1,), name="user_input")
movie_input = Input(shape=(1,), name="movie_input")

user_embedding = Embedding(
    input_dim=n_users, output_dim=embedding_size, name="user_embedding"
)(user_input)
movie_embedding = Embedding(
    input_dim=n_movies, output_dim=embedding_size, name="movie_embedding"
)(movie_input)

user_flatten = Flatten()(user_embedding)
movie_flatten = Flatten()(movie_embedding)

concatenated = Concatenate()([user_flatten, movie_flatten])
dense_layer_1 = Dense(64, activation="relu")(concatenated)
output_layer = Dense(1, activation="linear")(dense_layer_1)

# Compile the model
model = Model(inputs=[user_input, movie_input], outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mean_squared_error")

epochs = 1  # change the no of epochs accordingly.
# Model Training
history = model.fit(
    [train_data["userId"], train_data["movieId"]],
    train_data["rating"],
    epochs=epochs,
    batch_size=64,
    validation_split=0.2,
)

# Model Evaluation
predictions = model.predict([test_data["userId"], test_data["movieId"]])
rmse = np.sqrt(mean_squared_error(test_data["rating"], predictions.flatten()))
print(f"Deep Learning RMSE: {rmse}")
