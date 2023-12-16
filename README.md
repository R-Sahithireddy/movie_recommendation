# END TO END MOVIE RECOMMENDATION SYSTEM (Collaborative Filtering System)

## Table of Contents
- [Problem statement](#problem-statement)<br>
- [Description](#description)<br>
- [EDA Conclusion](#exploratory-data-analysis-eda-conclusion)<br>
- [Model Implementation](#model-implementation)<br>
- [Model Evaluation](#model-evaluation)<br>
- [Project Structure](#project-structure)<br>
- [References](#references)<br>

## Problem Statement: 
The problem revolves around predicting the top movies that we can recommend based on User and based on Item and design a Collaborative Filtering for Movie Recommendations and we use MovieLens dataset.<br>

## Description:
The objective of this project is to build a collaborative filtering recommendation system using the MovieLens dataset. Collaborative filtering is a technique that makes automatic predictions about the preferences of a user by collecting preferences from many users (collaborating). In this case, we'll be focusing on user-item and item-item collaborative filtering using the ratings given by users to movies.<br>

## Exploratory Data Analysis (EDA) Conclusion:

The EDA (Exploratory Data Analysis) on the MovieLens dataset (rating.csv) has provided valuable insights:

#### Data Volume: 
The dataset contains over 100k  entries. Due to computational constraints, we decide to train our model on a smaller fraction of the data for practicality and efficiency.

#### Timestamp Column: 
An additional column, timestamp, is present but not required for our collaborative filtering model. Thus, it is dropped to streamline the modeling process.

#### Null Values:
 No null values are present in the dataset, ensuring the quality of data for collaborative filtering model building.<br>

## Model Implementation:

The predictive model is implemented in the `src` folder, specifically in the `model.py` file. The model leverages functions for enhanced modularity and Various collaborative filtering techniques, including cosine similarity, are employed. Evaluation metrics, such as RMSE, are utilized to assess model performance.<br>
<br>
A deep learning-based collaborative filtering model is implemented in `deep_learning_model.py` within the `src` folder. This serves as a comparative analysis with traditional collaborative filtering techniques.

## Model Evaluation:

We predict the model's Performance and evaluate the model using Root Mean Squared Error(RMSE). <br>
The deep learning model did reasonably good with RMSE of 0.84 for a single epoch so there is a chance of better score of RMSE if we train the data with more number of epochs.

## Project Structure:
`src/EDA.ipynb`: Contains Exploratory Data Analysis(EDA) in Jupyter notebook.<br>
`src/model.py`: Contains the implemented predictive model using functions.<br>
`src/exceptions.py`: Exception handling function for reuse in the code.<br>
`src/deep_learning_model.py`: A deep learning-based collaborative filtering model is implemented in `deep_learning_model.py` within the src folder. This serves as a comparative analysis with traditional collaborative filtering techniques.<br>


## References:

Dataset Source(BOSTON HOUSING DATASET):  https://www.kaggle.com/datasets/abhikjha/movielens-100k.<br>
#### Notebooks Referred:
1. https://www.kaggle.com/code/mrisdal/starter-movielens-20m-dataset-144a8ee2-e<br>
2. https://github.com/krishnaik06/mlproject<br>
3. https://github.com/rposhala/Recommender-System-on-MovieLens-dataset<br>
4. https://github.com/asif536/Movie-Recommender-System<br>