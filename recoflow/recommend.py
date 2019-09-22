import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors 

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