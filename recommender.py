import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import re
from sklearn.feature_selection import r_regression
from tabulate import tabulate

def format_data(movies_path, ratings_path):
    #Reading CSVs
    ml_25m_movies_df = pd.read_csv(movies_path)
    ml_25m_rating_df = pd.read_csv(ratings_path)

    #Creating a frequency table
    num_ratings_df = pd.DataFrame()
    num_ratings_df['num_ratings'] = ml_25m_rating_df['movieId'].value_counts()
    num_ratings_df = num_ratings_df.reset_index()
    num_ratings_df = num_ratings_df.rename(columns={'index':'movieId'})

    #Creating year column
    ml_25m_movies_df['year'] = ml_25m_movies_df.title.str.extract('(\\d\d\d\d\))', expand=False)
    #Removing parentheses
    ml_25m_movies_df['year'] = ml_25m_movies_df.year.str.extract('(\d\d\d\d)',expand=False)
    #Removing year from title
    ml_25m_movies_df['title'] = ml_25m_movies_df["title"].apply(lambda x: re.sub("\(.*?\)", "", x))
    #Removing ending whitespace
    ml_25m_movies_df['title'] = ml_25m_movies_df['title'].apply(lambda x: x.strip())
    #Adding freq column
    ml_25m_movies_df = ml_25m_movies_df.merge(num_ratings_df, on='movieId', how='inner')

    return ml_25m_movies_df, ml_25m_rating_df

def recommend_movies(user_ratings_df, ml_25m_movies_df, ml_25m_rating_df, top_users_to_evaluate = 100, 
                     top_users_final = 50, min_year = 1970, min_ratings = 7500, min_weighted_rec_score = 0):
    #Getting ids of movies by title
    id = ml_25m_movies_df[ml_25m_movies_df['title'].isin(user_ratings_df['title'].tolist())]
    #Merging to get movieId. implicitly merged by title
    user_ratings_df = pd.merge(id, user_ratings_df)
    #getting users that have watched movies that the input user has watched and storing
    users = ml_25m_rating_df[ml_25m_rating_df['movieId'].isin(user_ratings_df['movieId'].tolist())]
    users_subset_group = users.groupby(['userId'])
    #Sorting so that users with movie ratings similar to user will have priority
    users_subset_group = sorted(users_subset_group, key=lambda x: len(x[1]), reverse=True)
    users_subset_group = users_subset_group[0:top_users_to_evaluate]

    #TODO
    correlation_dict = {}
    for user_id, group in users_subset_group:
        #Sorting user ratings and group so they are the same
        group = group.sort_values(by='movieId')
        user_ratings_df.sort_values(by='movieId')
        #Get ratings for movies in common
        temp = user_ratings_df[user_ratings_df['movieId'].isin(group['movieId'].tolist())]
        #Change to np arrays for sklearn
        temp_rating_list = np.array(temp['rating'].tolist())
        temp_group_list = np.array(group['rating'].tolist())
        # print(type(user_id))
        # print(user_id)
        # print(group)
        # print(users_subset_group)
        correlation_dict[user_id] = r_regression(temp_group_list.reshape(-1, 1), temp_rating_list)[0]
    
    #Creating a correlation dataframe from the correlation_dict
    correlation_df = pd.DataFrame.from_dict(correlation_dict, orient='index')
    correlation_df.columns = ['similarityIndex']
    correlation_df['userId'] = correlation_df.index
    correlation_df.index = range(len(correlation_df)) #reinstating the indicies
    #Get all movies similar users have watched
    top_similar_users = correlation_df.sort_values(by='similarityIndex', ascending=False)[0:top_users_final]
    top_similar_users_rating = top_similar_users.merge(ml_25m_rating_df, left_on='userId', right_on='userId', how='inner')
    #Getting a weighted rating using similarity
    top_similar_users_rating['weightedRating'] = top_similar_users_rating['similarityIndex']*top_similar_users_rating['rating']
    #Summing all weighted ratings and similarities per movie
    movie_scores = top_similar_users_rating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
    movie_scores.columns = ['sum_similarityIndex', 'sum_weightedRating']

    recommendation_df = pd.DataFrame()
    recommendation_df['weightedAverageRecommendationScore'] = movie_scores['sum_weightedRating']/movie_scores['sum_similarityIndex']
    recommendation_titles_df = recommendation_df.merge(ml_25m_movies_df, on='movieId', how='inner')

    #Cleaning up
    recommendation_titles_df = recommendation_titles_df.dropna()

    user_movies = user_ratings_df['title'].tolist()
    for movie in user_movies:
        recommendation_titles_df = recommendation_titles_df[recommendation_titles_df['title'] != movie]

    recommendation_titles_df = recommendation_titles_df[recommendation_titles_df['year'].astype(int) >= min_year]
    recommendation_titles_df = recommendation_titles_df[recommendation_titles_df['num_ratings'] >= min_ratings]
    recommendation_titles_df = recommendation_titles_df[recommendation_titles_df['weightedAverageRecommendationScore'] >= min_weighted_rec_score]

    recommendation_titles_df = recommendation_titles_df.sort_values(by='weightedAverageRecommendationScore', ascending=False)
    return recommendation_titles_df