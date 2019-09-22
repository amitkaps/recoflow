


def precision_at_k(rating_true, rating_pred, k):
    
    df_hit, df_hit_count, n_users = get_hit_df(rating_true, rating_pred, k)
    
    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / k).sum() / n_users


def recall_at_k(rating_true, rating_pred, k):

    df_hit, df_hit_count, n_users = get_hit_df(rating_true, rating_pred, k)

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users



def ndcg_at_k(rating_true, rating_pred, k):

    df_hit, df_hit_count, n_users = get_hit_df(rating_true, rating_pred, k)
    
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