import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def EncodeUserItem(df, user_col, item_col, rating_col, time_col):
    """Function to encode users and items
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be used.
        user_col (string): Name of the user column.
        item_col (string): Name of the item column.
        rating_col (string): Name of the rating column.
        timestamp_col (string): Name of the timestamp column.
    
    Returns: 
        encoded_df (pd.DataFrame): Modifed dataframe with the users and items index
        n_users (int): number of users
        n_items (int): number of items
        user_encoder (sklearn.LabelEncoder): Encoder for users.
        item_encoder (sklearn.LabelEncoder): Encoder for items.
    """
    
    interaction = df.copy()
    
    user_encoder = LabelEncoder()
    user_encoder.fit(interaction[user_col].values)
    n_users = len(user_encoder.classes_)
    
    item_encoder = LabelEncoder()
    item_encoder.fit(interaction[item_col].values)
    n_items = len(item_encoder.classes_)

    interaction["USER"] = user_encoder.transform(interaction[user_col])
    interaction["ITEM"] = item_encoder.transform(interaction[item_col])
    
    interaction.rename({rating_col: "RATING", time_col: "TIMESTAMP"}, axis=1, inplace=True)
    
    print("Number of users: ", n_users)
    print("Number of items: ", n_items)
    
    return interaction, n_users, n_items, user_encoder, item_encoder


def RandomSplit (df, ratios, shuffle=False):
    
    """Function to split pandas DataFrame into train, validation and test
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
    
    Returns: 
        list: List of pd.DataFrame split by the given specifications.
    """
    seed = 42                  # Set random seed
    if shuffle == True:
        df = df.sample(frac=1) # Shuffle the data
    samples = df.shape[0]      # Number of samples
    
    # Converts [0.7, 0.2, 0.1] to [0.7, 0.9]
    split_ratio = np.cumsum(ratios).tolist()[:-1] # Get split index
    
    # Get the rounded integer split index
    split_index = [round(x * samples) for x in split_ratio]
    
    # split the data
    splits = np.split(df, split_index)
    
    # Add split index (this makes splitting by group more efficient).
    for i in range(len(ratios)):
        splits[i]["split_index"] = i

    return splits


def _splitter (df, ratios, by="USER", chrono=False):
    
    """Function to split pandas DataFrame into train, validation and test (by user or item and in chronological order if needed)
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        by (string): split by USER or ITEM
        chrono (boolean): whether to sort in chronological order or not by TIMESTAMP
    
    Returns: 
        list: List of pd.DataFrame split by the given specifications.
    """
    seed = 42                  # Set random seed
    samples = df.shape[0]      # Number of samples
    col_time = "TIMESTAMP"
    col_user = "USER"
    col_item = "ITEM"
    
    # Split by each group and aggregate splits together.
    splits = []

    # Sort in chronological order, the split by user or item.      
    if chrono == True:
      if by == "USER":
        df_grouped = df.sort_values(col_time).groupby(col_user)
      else: 
        df_grouped = df.sort_values(col_time).groupby(col_item)
    
    # Split by user or item.      
    if chrono == False:
      if by == "USER":
        df_grouped = df.groupby(col_user)
      else: 
        df_grouped = df.groupby(col_item)
    
    for name, group in df_grouped:
        group_splits = RandomSplit(df_grouped.get_group(name), ratios, shuffle=False)
        
        # Concatenate the list of split dataframes.
        concat_group_splits = pd.concat(group_splits)
        splits.append(concat_group_splits)
    
    # Concatenate splits for all the groups together.
    splits_all = pd.concat(splits)

    # Take split by split_index
    splits_list = [ splits_all[splits_all["split_index"] == x] for x in range(len(ratios))]

    return splits_list

def ChronoSplit (df, ratios, by="USER"):
    
    """Function to split pandas DataFrame into train, validation and test (by user or item) in chronological order
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        by (string): split by USER or ITEM

    Returns: 
        list: List of pd.DataFrame split by the given specifications.
    """
    splits_list = _splitter(df, ratios, by, True)

    return splits_list


def StratifiedSplit (df, ratios, by="USER"):
    
    """Function to split pandas DataFrame into train, validation and test (by user or item)
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        by (string): split by USER or ITEM

    Returns: 
        list: List of pd.DataFrame split by the given specifications.
    """
    splits_list = _splitter(df, ratios, by, False)

    return splits_list


def GetGenre(items, item_encoder):
    cols = ['movie_id', 'genre_unknown', 'Action', 'Adventure',
       'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
       'Fantasy', 'FilmNoir', 'Horror', 'Musical', 'Mystery', 'Romance',
       'SciFi', 'Thriller', 'War', 'Western']
    df_genre = items[cols].copy()
    
    df_genre["ITEM"] = item_encoder.transform(df_genre.movie_id)
    df_genre["genre"] = df_genre[cols[1:]].apply(lambda x: ''.join(x.map(str)), axis=1)
    df_genre["genre_int"] = df_genre.genre.apply(int, args=(2,))
    df_genre.drop(cols[1:], axis=1, inplace=True)
    df_genre.drop_duplicates(inplace=True)
    df_genre.sort_values("ITEM", inplace=True)
    
    df_genre["genre_label"] = LabelEncoder().fit_transform(df_genre.genre)
    
    return df_genre