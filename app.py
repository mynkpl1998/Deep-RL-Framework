import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from dash.dependencies import Output,Input
import sys

exp_name = sys.argv[1]
interval = int(sys.argv[2])
window_size = int(sys.argv[3])

app = dash.Dash(__name__)


colors = {
    'align' : "center"
}

app.layout = html.Div(style={'textAlign':"center"},children=[
    html.H1(style={'textAlign': colors['align']}, children="Deep RL Framework : Training Statistics"),

    html.H2(children="A platform to visualize Deep RL Training "),

    dcc.Graph(id="live-action-stat"),
    dcc.Interval(id="interval-component", interval=int(interval)*1000, n_intervals=0),

    dcc.Graph(id="live-loss-stat"),
    dcc.Graph(id="live-reward-stat")
])

def moving_average(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

@app.callback(Output('live-action-stat','figure'),
    [Input('interval-component','n_intervals')])
def update_graphs(n):
    action_stat = pd.read_csv("data/"+exp_name+"_actions_stat.csv")
    #action_stat = action_stat.rolling(window_size).sum().fillna(0)

    trace1 = go.Bar(

        y = action_stat["Random"],
        x = np.arange(len(action_stat)),
        name = "Random"
    )

    trace2 = go.Bar(
        y = action_stat["Greedy"],
        x = np.arange(len(action_stat)),
        name = "Greedy"
    )
    figure = {
        'data' : [trace1,trace2],
        'layout' : go.Layout( title = "Action Distribution", xaxis = {'title':'Episodes'},yaxis = {'title':"Times Action Selected"}, font=dict(family='sans serif', size=12, color='black')
        )
    }

    return figure

@app.callback(Output('live-loss-stat','figure'),
    [Input('interval-component','n_intervals')])
def update_graphs_loss(n):
    loss_stat = pd.read_csv("data/"+exp_name+"_loss_stat.csv")
    loss_stat = loss_stat.rolling(window=window_size).mean()

    loss_trace = go.Scatter(
        y = loss_stat["loss"],
        x = np.arange(len(loss_stat)),
        name = "Loss"
    )

    figure = {
        'data' : [loss_trace],
        'layout': go.Layout( title = "Loss over frames",xaxis = {'title':'Steps/Frames'},yaxis = {'title':"Loss"}, font=dict(family='sans serif', size=12, color='black')
        )
        }

    return figure

@app.callback(Output('live-reward-stat','figure'),
    [Input('interval-component','n_intervals')])
def update_graphs_loss(n):
    total_reward_stat = pd.read_csv("data/"+exp_name+"_total_reward_stat.csv")
    total_reward_stat = total_reward_stat.rolling(window=window_size).mean()

    reward_trace = go.Scatter(
        y = total_reward_stat["total reward"],
        x = np.arange(len(total_reward_stat)),
        name = "Cumulative Reward"
    )

    figure = {
        'data' : [reward_trace],
        'layout' : go.Layout( title = "Total Reward over Episodes",xaxis = {'title':'Epsiode'},yaxis = {'title':"Episode Reward"}, font=dict(family='sans serif', size=12, color='black')
        )
        }

    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
