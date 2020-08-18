import numpy as np
import pandas as pd
from .utils import _UserItemCrossJoin, _FilterBy, _GetTopKItems, _GetHitDF

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL

def _GetEmbedding(model, name):
    """Function to get embedding for users or items
    
    Params:     
        model (keras model): Keras model which has embedding layer
        names (string): Name of the user or item embedding layer
    
    Returns: 
        embedding (numpy matrix): Embedding layer for user or item
    """
    embedding = model.get_layer(name = name).get_weights()[0]
    return embedding

def UserEmbedding(model, name="UserEmbedding"):
    """Function to get embedding for users
    
    Params:     
        model (keras model): Keras model which has embedding layer
        names (string): Name of the user embedding layer
    
    Returns: 
        embedding (numpy matrix): Embedding layer for users
    """
    return _GetEmbedding(model, name)


def ItemEmbedding(model, name="ItemEmbedding"):
    """Function to get embedding for items
    
    Params:     
        model (keras model): Keras model which has embedding layer
        names (string): Name of the item embedding layer
    
    Returns: 
        embedding (numpy matrix): Embedding layer for items
    """
    return _GetEmbedding(model, name)


def _GetSimilar(embedding, k):
    model_similar_items = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(embedding)
    distances, indices = model_similar_items.kneighbors(embedding)
    
    return distances, indices



def GetRankingTopK(model, data, train, k=5):
    """Get predictions for all users, removing train data

    Params:
        data (pandas.DataFrame): DataFrame of entire rating data
        train (pandas.DataFrame): DataFrame of train rating data
        k (int): number of items for each user

    Returns:
        pd.DataFrame: DataFrame of top k items for each user, sorted by `col_user` and `rank`
    
    """
    
    # Get predictions for all user-item combination
    all_predictions = GetPredictions(model, data)
    
    # Handle Missing Values
    all_predictions.fillna(0, inplace=True)
    
    # Filter already seen items
    all_predictions_unseen = _FilterBy(all_predictions, train, ["USER", "ITEM"])
    
    ranking_topk_df = _GetTopKItems(all_predictions_unseen, "USER", "RATING", k=5)
    
    return ranking_topk_df


def GetPredictions(model, data):
    """Get predictions for all user-item combinations
    
    Params:
        data (pandas.DataFrame): DataFrame of entire rating data
        model (Keras.model): Trained keras model
        
    Returns:
        pd.DataFrame: DataFrame of rating predictions for each user and each item
        
    """
    # Create the crossjoin for user-item
    user_item = _UserItemCrossJoin(data)
    
    # Score for every user-item combination
    user_item["RATING"] = model.predict([user_item.USER, user_item.ITEM])
    
    return user_item

def GetSimilar(embedding, k=5, metric="cosine"):
    """Get Similiar ITEMS or USER
    
    Params:
    embedding (np.array): Embedding for ITEM or USER
    k (int): No. of similar items to be presented
    metric (string): distance metrics to be used to find nearest neighbour
    
    Returns:
    indices (int): index of the nearest ITEM or USER
    
    """
    neighbors = k + 1
    
    model_similar_items = NearestNeighbors(n_neighbors=neighbors).fit(embedding)
    distances, indices = model_similar_items.kneighbors(embedding)
    
    return indices

def ShowSimilarItems(item_index, item_similar_indices, item_encoder, items, image_path="data/posters/"):
    """Show Similiar Items
    
    Params:
    item_index (int): Index of the Item
    item_similar_indices (np.array): Similar Item Indices
    item_encoder (sklearn.LabelEncoder): Label Encoder for Items
    items (pd.DataFrame): ITEMS DataFrame
    image_path (string): Relative path to image folder
    
    Returns:
    indices (int): index of the nearest ITEM or USER
    
    """
    movie_title = items.iloc[0].title
    
    s = item_similar_indices[item_index]
    movie_ids = item_encoder.inverse_transform(s)

    images = []
    titles = []
    for movie_id in movie_ids:
        img_path = image_path + str(movie_id) + '.jpg'
        images.append(mpimg.imread(img_path))
        title = items[items.movie_id == movie_id].title.tolist()[0]
        titles.append(title)

    plt.figure(figsize=(20,10))
    columns = 6
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.axis('off')
        plt.imshow(image)
        plt.title(titles[i])