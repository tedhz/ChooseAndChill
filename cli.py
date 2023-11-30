import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import recommender
from tabulate import tabulate
import pandas as pd
from textdistance import levenshtein
import os

#constants
MOVIES_PATH = os.getcwd() + "/ml_25m/movies.csv"
RATINGS_PATH = os.getcwd() + "/ml_25m/ratings.csv"

MAINSTREAM = 8000
POPULAR = 3000
INDIE = 1000
ALL = 0

def get_popularity():
    min_num_ratings = 0
    valid_popularity = False
    while not valid_popularity:
        min_popularity = input("Please enter one of the following popularity levels:\nMainstream\nPopular\nIndie\nAll\nEnter:\n")
        if(min_popularity.lower() == "mainstream"):
            min_num_ratings = MAINSTREAM
            valid_popularity = True
        elif(min_popularity.lower() == "popular"):
            min_num_ratings = POPULAR
            valid_popularity = True
        elif(min_popularity.lower() == "indie"):
            min_num_ratings = INDIE
            valid_popularity = True
        elif(min_popularity.lower() == "all"):
            min_num_ratings == ALL
            valid_popularity = True
        else:
            print("Invalid popularity level.")
    return min_num_ratings

def get_min_year():
    valid_year = False
    min_year = 0
    while not valid_year:
        min_year_str = input("Please enter a minimum year (Enter 0 for all years):\n")
        if min_year_str.isdigit():
            min_year = int(min_year_str)
            if(min_year >= 0):
                valid_year = True
            else:
                print("Please enter a postive year or 0.")
        else:
            print("Not an integer year. Please try again.")
    return min_year

def get_num_recommendations():
    num = 10
    valid_num = False
    while not valid_num:
        num_str = input("Please enter the number of recommendations you would like to see:\n")
        if num_str.isdigit():
            num = int(num_str)
            if num >= 1:
                valid_num = True
            else:
                print("Integer must be at least 1.")
        else:
            print("Not an integer.")
    return num


def cli():
    exit = False
    print("Welcome to Chill and Choose! A movie recommendation engine using collaborative filtering!")
    print("Loading data...")
    movies_df, ratings_df = recommender.format_data(MOVIES_PATH, RATINGS_PATH)

    while not exit:
        # user_ids = []
        user_movies = []
        user_ratings = []

        done_rating = False
        continue_str = input("Press any key to start movie selection. Enter /exit to end the program.\n")
        if continue_str == "/exit":
            exit = True
            continue

        while not done_rating:
            #TODO: implement suggestion and year

            title = input('Input Movie (before 2019) Title (type /finished to stop):\n')       
            if title == "/finished":
                done_rating = True
            else:
                search_result = movies_df.loc[movies_df['title'].str.contains(title, na=False, case=False)] #use levenshtein distance
                if search_result.empty:
                    print("There were no movies that matched your search, please try again.")
                elif(len(search_result) > 1):
                    print("Here's what we found:")
                    print(tabulate(search_result.head(10).drop('num_ratings', axis=1), headers='keys', tablefmt='psql'))
                    print("Please enter the movie title exactly as shown.")
                else:
                    user_movies.append(search_result.iloc[0]['title'])

                    #Asking user to rate the chosen movie
                    valid_rating = False
                    while not valid_rating:
                        rating_str = input("Please enter an integer rating from 1-5 for the chosen movie:\n")
                        if rating_str.isdigit():
                            rating_int = int(rating_str)
                            if (rating_int >= 1) and (rating_int <= 5):
                                user_ratings.append(rating_int)
                                valid_rating = True
                            else:
                                print("Invalid number. Please try again.")
                        else:
                            print("Not an integer. Please try again.")
        #End of choosing movies and ratings

        #Ask user for filtering parameters
        #Asking user for minimum year
        min_year = get_min_year()
        #Asking user for popularity:
        min_num_ratings = get_popularity()
        #Getting number of recs
        num_recommendations = get_num_recommendations()

        #Converting lists to dfs
        user_ratings_dict = {'title': user_movies, 'rating': user_ratings}
        user_ratings_df = pd.DataFrame(user_ratings_dict)

        recommendation_df = recommender.recommend_movies(user_ratings_df=user_ratings_df, ml_25m_movies_df=movies_df, ml_25m_rating_df=ratings_df, min_year=min_year, min_ratings=min_num_ratings)
        recommendation_df = recommendation_df.drop('weightedAverageRecommendationScore', axis=1)
        recommendation_df = recommendation_df.drop('num_ratings', axis=1)
        recommendation_df = recommendation_df.drop('movieId', axis=1)

        print(f"Here are your top {num_recommendations} recommendations")
        print(tabulate(recommendation_df.head(num_recommendations).reset_index(drop=True), headers='keys', tablefmt='psql'))

cli()
