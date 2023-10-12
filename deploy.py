import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns

import plotly.express as px

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import pandas as pd

import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import plotly.express as px
from jupyter_dash import JupyterDash

import plotly.graph_objects as go

csv_url = "https://raw.githubusercontent.com/varun0087/deploy/main/Mall_Customers.csv"

# Read the CSV file into a Pandas DataFrame, skipping rows with errors
df = pd.read_csv(csv_url, error_bad_lines=False)

#droping columns with categorical features
df.select_dtypes('object')
#df.drop(columns='private', inplace=True,axis=1)
#top 10 variance features
top_10=df.var().sort_values().tail(10)

features=top_10.tail().index.to_list()
X=df[features]



app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server=app.server


app.layout=html.Div([
    
    html.H1("K-Means_Clustering",style={"textAlign":'center',"color":'blue'}),
    html.Br(),
    html.Br(),
    
    dbc.Row([
        
        dbc.Col([
            dbc.Row([
                dbc.Label("Select X-variable",style={"marginLeft":'10px','padding':'20px'}),
                dcc.Dropdown(id='x1',options=[
                        {"label": col, "value": col} for col in X.columns
                    ],value=features[0],style={"marginLeft":'10px','padding':'10px'})
            ]),
            html.Br(),
            dbc.Row([
                dbc.Label("Select Y-variable",style={"marginLeft":'10px','padding':'20px'}),
                dcc.Dropdown(id='x2',options=[
                        {"label": col, "value": col} for col in X.columns
                    ],value=features[1],style={"marginLeft":'10px','padding':'10px'})
            ]),
            html.Br(),
            dbc.Row([
                dbc.Label("Select n_Clusters",style={"marginLeft":'10px'}),
                dcc.Input(id='num',min=2,max=15,step=1,value=2,type='number',style={"marginLeft":'10px','padding':'10px'})
            ]),
            
            
            
            
        ],width=3),
        
        dbc.Col([
            
            dcc.Graph(id='figu')
            
        ],width=8),
        
        
    ],),
    
    
    
    
    
    
    
])


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@app.callback(Output('figu','figure'),
             [Input('x1','value'),
             Input('x2','value'),
             Input('num','value')])
def make_graph(x, y, n_clusters):
    # minimal input validation, make sure there's at least one cluster
    km = KMeans(n_clusters=n_clusters)
    
    km.fit(X)
    X["cluster"] = km.labels_

    centers = km.cluster_centers_

    data = [
        go.Scatter(
            x=X.loc[X.cluster == c, x],
            y=X.loc[X.cluster == c, y],
            mode="markers",
        
            marker={"size": 8},
            name="Cluster {}".format(c),
        )
        for c in range(n_clusters)
    ]

    data.append(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker={"color": "#000", "size": 12, "symbol": "diamond"},
            name="Cluster centers",
        )
    )

    layout = {"xaxis": {"title": x}, "yaxis": {"title": y}}
    fig=go.Figure(data=data, layout=layout)
    fig.update_layout(plot_bgcolor="white")

    return fig



if __name__ == "__main__":
    app.run_server()
