import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .utils import _UserItemCrossJoin, _FilterBy, _GetTopKItems, _GetHitDF, _MergeRatingTruePred


def MeanSquaredError(rating_true, rating_pred):
  """Calculate Mean Squared Error
  
  Params:
  rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
  rating_pred (pd.DataFrame): Predicated Ratings.

  Returns:
  float: Mean Squared Error
  """

  y_true, y_pred = _MergeRatingTruePred(
    rating_true=rating_true, 
    rating_pred=rating_pred)
  
  return mean_squared_error(y_true, y_pred)


def RootMeanSquaredError(rating_true, rating_pred):
  """Calculate Root Mean Squared Error
  
  Params:
  rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
  rating_pred (pd.DataFrame): Predicated Ratings.

  Returns:
  float: Root Mean Squared Error
  """

  y_true, y_pred = _MergeRatingTruePred(
    rating_true=rating_true, 
    rating_pred=rating_pred)
  
  return np.sqrt(mean_squared_error(y_true, y_pred))


def MeanAbsoluteError(rating_true, rating_pred):
  """Calculate Mean Absolute Error
  
  Params:
  rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
  rating_pred (pd.DataFrame): Predicated Ratings.

  Returns:
  float: Mean Absolute Error
  """

  y_true, y_pred = _MergeRatingTruePred(
    rating_true=rating_true, 
    rating_pred=rating_pred)
  
  return mean_absolute_error(y_true, y_pred)


def PrecisionK(rating_true, rating_pred, k=5):
    """Calculate Precision at K
    
    Params:
    rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
    rating_pred (pd.DataFrame): Predicated Ratings.
    k (int): Number of items presented

    Returns:
    float: Precision at K
    """

    df_hit, df_hit_count, n_users = _GetHitDF(rating_true, rating_pred, k)
    
    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / k).sum() / n_users


def RecallK(rating_true, rating_pred, k=5):
    """Calculate Precision at K
    
    Params:
    rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
    rating_pred (pd.DataFrame): Predicated Ratings.
    k (int): Number of items presented

    Returns:
    float: Recall at K
    """

    df_hit, df_hit_count, n_users = _GetHitDF(rating_true, rating_pred, k)

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users



def NDCGK(rating_true, rating_pred, k=5):
    """Calculate NDCG at K
    
    Params:
    rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
    rating_pred (pd.DataFrame): Predicated Ratings.
    k (int): Number of items presented

    Returns:
    float: NDCG at K
    """
    df_hit, df_hit_count, n_users = _GetHitDF(rating_true, rating_pred, k)
    
    if df_hit.shape[0] == 0:
        return 0.0

    # calculate discounted gain for hit items
    df_dcg = df_hit.copy()
    # relevance in this case is always 1
    df_dcg["dcg"] = 1 / np.log1p(df_dcg["rank"])
    # sum up discount gained to get discount cumulative gain
    df_dcg = df_dcg.groupby("USER", as_index=False, sort=False).agg({"dcg": "sum"})
    # calculate ideal discounted cumulative gain
    df_ndcg = pd.merge(df_dcg, df_hit_count, on=["USER"])
    df_ndcg["idcg"] = df_ndcg["actual"].apply(
        lambda x: sum(1 / np.log1p(range(1, min(x, k) + 1)))
    )

    # DCG over IDCG is the normalized DCG
    return (df_ndcg["dcg"] / df_ndcg["idcg"]).sum() / n_users


def MeanAveragePrecisionK(rating_true, rating_pred, k=5):
    """Calculate Mean Average Precision at K
    
    Params:
    rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
    rating_pred (pd.DataFrame): Predicated Ratings.
    k (int): Number of items presented

    Returns:
    float: Mean Average Precision at K
    """
    df_hit, df_hit_count, n_users = _GetHitDF(rating_true, rating_pred, k)
    
    if df_hit.shape[0] == 0:
        return 0.0

    # Calculate Reciprocal Rank
    df_hit_sorted = df_hit.copy()
    df_hit_sorted["rRank"] = (df_hit_sorted.groupby("USER").cumcount() + 1) / df_hit_sorted["rank"]
    df_hit_sorted = df_hit_sorted.groupby("USER").agg({"rRank": "sum"}).reset_index()

    # Calculate Mean Averate Precision
    df_merge = pd.merge(df_hit_sorted, df_hit_count, on="USER")
    return (df_merge["rRank"] / df_merge["actual"]).sum() / n_users

def MeanReciprocalRankK(rating_true, rating_pred, k=5):
    """Calculate Mean Reciprocal Rank at K
    
    Params:
    rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
    rating_pred (pd.DataFrame): Predicated Ratings.
    k (int): Number of items presented

    Returns:
    float: Mean Reciprocal Rank at K
    """
    df_hit, df_hit_count, n_users = _GetHitDF(rating_true, rating_pred, k)
    
    if df_hit.shape[0] == 0:
        return 0.0

    # Calculate Reciprocal Rank
    df_hit_sorted = df_hit.copy()
    df_hit_sorted["rRank"] = (df_hit_sorted.groupby("USER").cumcount() + 1) / df_hit_sorted["rank"]
    df_hit_sorted = df_hit_sorted.groupby("USER").agg({"rRank": "sum"}).reset_index()

    return df_hit_sorted["rRank"].sum() / (n_users * k)


def RatingMetrics(rating_true, rating_pred):
    """Get all rating metrics
    
    Params:
    rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
    rating_pred (pd.DataFrame): Predicated Ratings.
    
    Returns:
    rating_metrics (pd.DataFrame): Ratings metrics MAE, MSE, RMSE
    """
    
    MSE = MeanSquaredError(rating_true, rating_pred)
    RMSE = RootMeanSquaredError(rating_true, rating_pred)
    MAE = MeanAbsoluteError(rating_true, rating_pred)
    
    rating_metrics = pd.DataFrame({
        "MSE": [float("{0:.4f}".format(MSE))],
        "RMSE": [float("{0:.4f}".format(RMSE))],
        "MAE": [float("{0:.4f}".format(MAE))]
    })
    
    return rating_metrics


def RankingMetrics(rating_true, rating_pred, k=5):
    """Get all rating metrics
    
    Params:
    rating_true (pd.DataFrame): Ground Truth Ratings. There should be no duplicate
    rating_pred (pd.DataFrame): Predicated Ratings.
    k (int): Number of items presented
    
    Returns:
    ranking_metrics (pd.DataFrame): Ranking metrics MAE, MSE, RMSE
    """
    
    Precision = PrecisionK(rating_true, rating_pred, k)
    Recall = RecallK(rating_true, rating_pred, k)
    MAP = MeanAveragePrecisionK(rating_true, rating_pred, k)
    MRR = MeanReciprocalRankK(rating_true, rating_pred, k)
    NDCG = NDCGK(rating_true, rating_pred, k)
    
    ranking_metrics = pd.DataFrame({
        "k": [k],
        "Precision@k": [float("{0:.4f}".format(Precision))],
        "Recall@k": [float("{0:.4f}".format(Recall))],
        "MAP@k": [float("{0:.4f}".format(MAP))],
        "MRR@k": [float("{0:.4f}".format(MRR))],
        "NDCG@k": [float("{0:.4f}".format(NDCG))]
    })
    
    return ranking_metrics