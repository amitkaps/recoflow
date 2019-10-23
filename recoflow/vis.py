import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot a 3d 
def Vis3d(X,Y,Z):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='y')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# Visualise the metrics from the model
def MetricsVis(history):
    df = pd.DataFrame(history)
    df.reset_index()
    df["batch"] = df.index + 1
    df = df.melt("batch", var_name="name")
    df["val"] = df.name.str.startswith("val")
    df["type"] = df["val"]
    df["metrics"] = df["val"]
    df.loc[df.val == False, "type"] = "training"
    df.loc[df.val == True, "type"] = "validation"
    df.loc[df.val == False, "metrics"] = df.name
    df.loc[df.val == True, "metrics"] = df.name.str.split("val_", expand=True)[1]
    df = df.drop(["name", "val"], axis=1)
    
    base = alt.Chart().encode(
        x = "batch:Q",
        y = "value:Q",
        color = "type"
    ).properties(width = 300, height = 300)

    layers = base.mark_circle(size = 50).encode(tooltip = ["batch", "value"]) + base.mark_line()
    chart = layers.facet(column='metrics:N', data=df).resolve_scale(y='independent')    
    
    return chart


def InteractionVis(df):
    
    vis = alt.Chart(df).mark_rect().encode(
    alt.X(field="ITEM", type="nominal",
         axis=alt.Axis(orient="top", labelAngle=0)),
    alt.Y(field="USER", type="nominal",
         axis=alt.Axis(orient="left")),
    alt.Color(field="RATING", type="quantitative", 
              scale=alt.Scale(type="bin-ordinal", scheme='yellowgreenblue', nice=True),
              legend=alt.Legend(titleOrient='top', orient="bottom", 
                                direction= "horizontal", tickCount=5))
    ).properties(
        width= 180,
        height=300
    ).configure_axis(
      grid=False
    )
    
    return vis


def TrainTestVis(train, test):
    
    df = pd.concat([train, test])
    maptt = {0: "train", 1: "test"}
    df["SPLIT"] = df.split_index.apply(lambda x: maptt[x])
    df.head()
    
    vis = alt.Chart(df).mark_rect().encode(
    alt.X(field="ITEM", type="nominal",
         axis=alt.Axis(orient="top", labelAngle=0)),
    alt.Y(field="USER", type="nominal",
         axis=alt.Axis(orient="left")),
    alt.Color(field="SPLIT", type="ordinal", 
              scale=alt.Scale(type="ordinal", scheme="darkred", nice=True),
              legend=alt.Legend(titleOrient='top', orient="bottom", 
                                direction= "horizontal", tickCount=5)),
    alt.Opacity(value=1)
    ).properties(
        width= 180,
        height=300
    ).configure_axis(
      grid=False
    )

    return vis

def EmbeddingVis(embedding, n_factors, name):
    embedding_df_wide = pd.DataFrame(embedding)
    embedding_df_wide[name]= embedding_df_wide.index
    embedding_df = pd.melt(embedding_df_wide, id_vars=[name], value_vars=np.arange(n_factors).tolist(),
       var_name='dim', value_name='value')
    
    dim = n_factors
    
    if name == "ITEM":
        vis = alt.Chart(embedding_df).mark_rect().encode(
            alt.X(field=name, type="nominal", axis=alt.Axis(orient="top", labelAngle=0)),
            alt.Y(field="dim", type="nominal", axis=alt.Axis(orient="left")),
            alt.Color(field="value", type="quantitative", 
                  scale=alt.Scale(type="bin-ordinal", scheme='yellowgreenblue', nice=True),
                  legend=alt.Legend(titleOrient='top', orient="bottom", 
                                direction= "horizontal", tickCount=5))
            ).properties(
                width=180,
                height=30*dim
            )
    
    else:
        vis = alt.Chart(embedding_df).mark_rect().encode(
            alt.X(field="dim", type="nominal", axis=alt.Axis(orient="top", labelAngle=0)),
            alt.Y(field=name, type="nominal", axis=alt.Axis(orient="left")),
            alt.Color(field="value", type="quantitative", 
                  scale=alt.Scale(type="bin-ordinal", scheme='yellowgreenblue', nice=True),
                  legend=alt.Legend(titleOrient='top', orient="bottom", 
                                direction= "horizontal", tickCount=5))
            ).properties(
                width=30*dim,
                height=300
            )
    
    
    return vis

def SimilarityVis(item_embedding, user_embedding):
    
    item_embedding_df_wide = pd.DataFrame(item_embedding)
    user_embedding_df_wide = pd.DataFrame(user_embedding)

    item_embedding_df_wide.reset_index(inplace=True)
    item_embedding_df_wide["idx"] = item_embedding_df_wide["index"].apply(lambda x: "I" + str(x))
    item_embedding_df_wide.columns = ["index", "X0", "X1", "idx" ]

    user_embedding_df_wide.reset_index(inplace=True)
    user_embedding_df_wide["idx"] = user_embedding_df_wide["index"].apply(lambda x: "U" + str(x))
    user_embedding_df_wide.columns = ["index", "X0", "X1", "idx" ]

    embedding_df_wide = pd.concat([item_embedding_df_wide, user_embedding_df_wide])
    
    base = alt.Chart(embedding_df_wide).encode(
        alt.X("X0:Q", axis = alt.Axis(bandPosition = 0.5)),
        alt.Y("X1:Q", axis = alt.Axis(bandPosition = 0.5))
    )

    vis = base.mark_point(size=0) + base.mark_text().encode(text="idx")
    
    return vis