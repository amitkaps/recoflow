import numpy as np
import pandas as pd
import requests, zipfile, io

# Get the url for ml-100k
zip_file_url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Get the files and extract them
# print("Downloading movielens 100k data...")
# r = requests.get(zip_file_url, stream=True)
# z = zipfile.ZipFile(io.BytesIO(r.content))
# z.extractall()
# print("Download and Extracted.")


# User Data 
# u.user     -- Demographic information about the users; this is a tab
#              separated list of
#              user id | age | gender | occupation | zip code
#              The user ids are the ones used in the u.data data set.

users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')


# Item from the README
# u.item     -- Information about the items (movies); this is a tab separated
#               list of
#               movie id | movie title | release date | video release date |
#               IMDb URL | unknown | Action | Adventure | Animation |
#               Children's | Comedy | Crime | Documentary | Drama | Fantasy |
#               Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
#               Thriller | War | Western |
#               The last 19 fields are the genres, a 1 indicates the movie
#               is of that genre, a 0 indicates it is not; movies can be in
#               several genres at once.
#               The movie ids are the ones used in the u.data data set.
genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
    "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"
]
items_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
items = pd.read_csv('ml-100k/u.item', sep='|', names=items_cols, encoding='latin-1')



# Ratings from the README
# u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
#               Each user has rated at least 20 movies.  Users and items are
#               numbered consecutively from 1.  The data is randomly
#               ordered. This is a tab separated list of 
#               user id | item id | rating | timestamp. 
#               The time stamps are unix seconds since 1/1/1970 UTC
ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

# Memory Usage
print(users.memory_usage(index=False, deep=True))


users.to_csv("users.csv.gz", index=False, compression="gzip")
items.to_csv("items.csv.gz", index=False, compression="gzip")
ratings.to_csv("ratings.csv.gz", index=False, compression="gzip")


