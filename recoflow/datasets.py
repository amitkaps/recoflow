import numpy as np 
import pandas as pd 


def SampleData(users, items, ratings):
    sample_movie_id = [1, 71, 95, 50, 176, 82]
    sample_user_id = [1, 2, 6, 7, 8, 10, 12, 13, 14, 16]
    
    sample_ratings = ratings[ratings.user_id.isin(sample_user_id) & ratings.movie_id.isin(sample_movie_id)]
    sample_items = items[items.movie_id.isin(sample_movie_id)]
    sample_users = users[users.user_id.isin(sample_user_id)]
    
    return sample_users, sample_items, sample_ratings
    
def SampleEvaluate():
    df_true = pd.DataFrame(
        {
            "USER": [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "ITEM": [1, 2, 3, 1, 4, 5, 6, 7, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14],
            "RATING": [5, 4, 3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1],
        }
    )

    df_pred = pd.DataFrame(
        {
            "USER": [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "ITEM": [3, 10, 12, 10, 3, 11, 5, 13, 4, 10, 7, 13, 1, 3, 5, 2, 11, 14],
            "RATING": [14, 13, 12, 14, 13, 12, 11, 10, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
        }
    )

    return df_true, df_pred


