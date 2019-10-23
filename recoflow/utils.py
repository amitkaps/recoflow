import numpy as np
import pandas as pd
import os, time, sys, math


def CreateDirectory(directory_path):
    """
    Checks whether a directory exists in the current path, and if not creates it.
    
    directory_path: path string for the folder (relative to current working directory)
    """
    
    # Get current path
    current_path = os.getcwd()
    current_path
    
    # define the name of the directory to be created
    new_dir_path= current_path + "/" + directory_path
    
    # Check if feature dir exists
    if os.path.exists(new_dir_path):
        print("Directory already exists %s" % new_dir_path)
    else:     
        try:
            os.mkdir(new_dir_path)
        except OSError:
            print("Creation of the directory %s failed" % new_dir_path)
        else:
            print("Successfully created directory %s" % new_dir_path)

def _MergeRatingTruePred(rating_true, rating_pred):
  """Joins ground truth and predictions data frames on USER and ITEMS

  Params:
  rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
  rating_pred (pd.DataFrame): Predicated Ratings.

  Returns:
  np.array: Array with true ratings
  np.array: Array with predicted ratings
  """
  suffixes = ["_TRUE", "_PRED"]
  rating_true_pred = pd.merge(rating_true, rating_pred, on=["USER", "ITEM"], suffixes=suffixes)

  return rating_true_pred["RATING_TRUE"], rating_true_pred["RATING_PRED"]


def _UserItemCrossJoin(df):
    """
    Get cross-join of all users and items
    
    Args:
        df (pd.DataFrame): Source dataframe.

    Returns:
        pd.DataFrame: Dataframe with crossjoins
    
    """
    
    crossjoin_list = []
    for user in df.USER.unique():
        for item in df.ITEM.unique():
            crossjoin_list.append([user, item])

    cross_join_df = pd.DataFrame(data=crossjoin_list, columns=["USER", "ITEM"])
    
    return cross_join_df
    

def _FilterBy(df, filter_by_df, filter_by_cols):
    """From the input DataFrame (df), remove the records whose target column (filter_by_cols) values are
    exist in the filter-by DataFrame (filter_by_df)

    Args:
        df (pd.DataFrame): Source dataframe.
        filter_by_df (pd.DataFrame): Filter dataframe.
        filter_by_cols (iterable of str): Filter columns.

    Returns:
        pd.DataFrame: Dataframe filtered by filter_by_df on filter_by_cols
    """

    return df.loc[
        ~df.set_index(filter_by_cols).index.isin(
            filter_by_df.set_index(filter_by_cols).index
        )
    ]


def _GetTopKItems(df, col_user, col_rating, k):
    """Get the top k items for each user.

    Params:
        dataframe (pandas.DataFrame): DataFrame of rating data
        col_user (str): column name for user
        col_rating (str): column name for rating
        k (int): number of items for each user

    Returns:
        pd.DataFrame: DataFrame of top k items for each user, sorted by `col_user` and `rank`
    """
    # Sort dataframe by col_user and (top k) col_rating
    top_k_items = (
        df.groupby(col_user, as_index=False)
        .apply(lambda x: x.nlargest(k, col_rating))
        .reset_index(drop=True)
    )
    # Add ranks
    top_k_items["rank"] = top_k_items.groupby(col_user, sort=False).cumcount() + 1
    return top_k_items




def _GetHitDF(rating_true, rating_pred, k):
    """Get Hit defined by relevancy, a hit usually means whether the recommended "k" items hit the "relevant" items by the user.
    
    Params:
    rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
    rating_pred (pd.DataFrame): Predicated Ratings.
    k (int): Number of items presented

    Returns:
    pd.DataFrame: Whether recommended K items hit the relevant item by user
    """    


    # Make sure the prediction and true data frames have the same set of users
    common_users = set(rating_true["USER"]).intersection(set(rating_pred["USER"]))
    rating_true_common = rating_true[rating_true["USER"].isin(common_users)]
    rating_pred_common = rating_pred[rating_pred["USER"].isin(common_users)]
    n_users = len(common_users)

    df_hit = _GetTopKItems(rating_pred_common, "USER", "RATING", k)
    df_hit = pd.merge(df_hit, rating_true_common, on=["USER", "ITEM"])[
        ["USER", "ITEM", "rank"]
    ]

    # count the number of hits vs actual relevant items per user
    df_hit_count = pd.merge(
        df_hit.groupby("USER", as_index=False)["USER"].agg({"hit": "count"}),
        rating_true_common.groupby("USER", as_index=False)["USER"].agg(
            {"actual": "count"}
        ),
        on="USER",
    )
    
    return df_hit, df_hit_count, n_users

def NegativeSamples(df, rating_threshold, ratio_neg_per_user=1):
    """ function to sample negative feedback from user-item interaction dataset.

    This negative sampling function will take the user-item interaction data to create 
    binarized feedback, i.e., 1 and 0 indicate positive and negative feedback, 
    respectively. 

    Args:
        df (pandas.DataFrame): input data that contains user-item tuples.
        rating_threshold (int): value below which feedback is set to 0 and above which feedback is set to 1
        ratio_neg_per_user (int): ratio of negative feedback w.r.t to the number of positive feedback for each user. 

    Returns:
        pandas.DataFrame: data with negative feedback 
    """
    
    df.columns = ["USER", "ITEM", "RATING", "unix_timestamp"]
    
    seed = 42
    
    df_pos = df.copy()
    df_pos["RATING"] = df_pos["RATING"].apply(lambda x: 1 if x >= rating_threshold else 0)
    df_pos = df_pos[df_pos.RATING>0]


    # Create a dataframe for all user-item pairs 
    df_neg = _UserItemCrossJoin(df)

    #remove positive samples from the cross-join dataframe
    df_neg = _FilterBy(df_neg, df_pos, ["USER", "ITEM"])    

    #Add a column for rating - setting it to 0
    df_neg["RATING"] = 0
   
    # Combine positive and negative samples into a single dataframe
    df_all = pd.concat([df_pos, df_neg], ignore_index=True, sort=True)
    df_all = df_all[["USER", "ITEM", "RATING"]]
    
    
    # Sample negative feedback from the combined dataframe.
    df_sample = (
        df_all.groupby("USER")
        .apply(
            lambda x: pd.concat(
                [
                    x[x["RATING"] == 1],
                    x[x["RATING"] == 0].sample(
                        min(
                            max(
                                round(len(x[x["RATING"] == 1]) * ratio_neg_per_user), 1
                            ),
                            len(x[x["RATING"] == 0]),
                        ),
                        random_state=seed,
                        replace=False,
                    )
                    if len(x[x["RATING"] == 0] > 0)
                    else pd.DataFrame({}, columns=["USER", "ITEM", "RATING"]),
                ],
                ignore_index=True,
                sort=True,
            )
        )
        .reset_index(drop=True)
        .sort_values("USER")
    )

    df_sample.columns = ["movie_id", "rating", "user_id"]
    return df_sample[["user_id", "movie_id", "rating"]]
