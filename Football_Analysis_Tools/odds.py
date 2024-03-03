import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import urllib3
from scipy.stats import poisson

from random import randint
from statistics import mode
from PIL import Image
from tabulate import tabulate

import seaborn as sns


def plot_players_xg_dis(ax, team_id, color, data, min_player_count=1):
    # Get xg distributions from dataframes
    all_shooters = data.groupby(['playerName'])['xG'].sum()
    all_shooters = all_shooters[all_shooters >= 0.15]

    plot_shooters = data[data['teamId']==team_id]
    plot_shooters = plot_shooters.groupby(['playerName'])['xG'].sum()
    plot_shooters = plot_shooters[plot_shooters >= 0.15]

    # Scale both teams by maximum value in all_shooters
    scale = all_shooters.max()
    plot_shooters_scaled = plot_shooters / scale
    plot_shooters_scaled.index = [name.split()[-1] for name in plot_shooters_scaled.index]

    x = plot_shooters_scaled.index
    y = plot_shooters_scaled.values

    # Add player xG values to the bar plot
    for i, val in enumerate(plot_shooters):
        ax.text(i, plot_shooters_scaled[i] + 0.05, round(val, 2), color='#E1D3D3', fontweight='bold', ha='center', fontsize=8)


    # bar_width = 0.2 if len(x) < min_player_count else 1.0
    ax.bar(x, y, color=color, width=.2, edgecolor='#8d99ae')

    ax.set_xlabel('Player',fontsize=8, color='#E1D3D3')
    ax.set_ylim([0, 1.1])  # Set y-axis limit to 1.1 for scaling
    ax.tick_params(axis='x', colors='#E1D3D3')
    ax.tick_params(axis='y', colors='#E1D3D3')
    ax.set_yticks([])
    ax.set_ylabel('')

    # Limit x-axis range to adjust spacing between bars
    if len(x) >= min_player_count:
        ax.set_xlim([-0.5, len(x)-0.5])

    return ax






#%%





def simulate_match_on_shots(shot_df):
    '''
    This function takes a match ID and simulates an outcome based on the shots
    taken by each team.
    '''

    shots = shot_df.copy()

    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']

    home_goals = 0
    if shots_home['xG'].shape[0] > 0:
        for shot in shots_home['xG']:
            # Sample a number from the Poisson distribution using the xG value as the lambda parameter
            goals = poisson.rvs(mu=shot)
            home_goals += goals

    away_goals = 0
    if shots_away['xG'].shape[0] > 0:
        for shot in shots_away['xG']:
            # Sample a number from the Poisson distribution using the xG value as the lambda parameter
            goals = poisson.rvs(mu=shot)
            away_goals += goals

    return {'home_goals':home_goals, 'away_goals':away_goals}


def iterate_k_simulations_on_match_id(shot_df, k=10000):
    '''
    Performs k simulations on a match, and returns the probabilites of a win, loss, draw.
    '''
    # Count the number of occurances
    home = 0
    draw = 0
    away = 0

    for i in range(k):
        simulation = simulate_match_on_shots(shot_df)
        if simulation['home_goals'] > simulation['away_goals']:
            home += 1
        elif simulation['home_goals'] < simulation['away_goals']:
            away += 1
        else:
            draw += 1
    home_prob = home / k
    draw_prob = draw / k
    away_prob = away / k
    return {'home_prob': home_prob, 'away_prob': away_prob, 'draw_prob': draw_prob}




def simulate_and_plot_match_result(ax,colors,shot_df, k=10000):

    shot_df = shot_df[shot_df['isOwnGoal']==0]

    Fotmob_home_name = shot_df[shot_df['Venue'] == 'Home']
    Fotmob_away_name = shot_df[shot_df['Venue'] == 'Away']
    Fotmob_home_name = Fotmob_home_name['TeamName'].iloc[0]
    Fotmob_away_name = Fotmob_away_name['TeamName'].iloc[0]

    def iterate_k_simulations_on_match_id(shot_df, k=10000):
        '''
        Performs k simulations on a match, and returns the probabilites of a win, loss, draw.
        '''
        # Count the number of occurances
        home = 0
        draw = 0
        away = 0

        for i in range(k):
            simulation = simulate_match_on_shots(shot_df)
            if simulation['home_goals'] > simulation['away_goals']:
                home += 1
            elif simulation['home_goals'] < simulation['away_goals']:
                away += 1
            else:
                draw += 1
        home_prob = home / k
        draw_prob = draw / k
        away_prob = away / k
        return {'home_prob': home_prob, 'away_prob': away_prob, 'draw_prob': draw_prob}



    result = iterate_k_simulations_on_match_id(shot_df)



    team_probs = {Fotmob_home_name: result['home_prob'], 'Draw': result['draw_prob'], Fotmob_away_name: result['away_prob']}
    labels = [Fotmob_home_name, 'Draw', Fotmob_away_name]


    # Create a horizontal bar plot
    ax.barh(labels, [result['home_prob'], result['draw_prob'], result['away_prob']],height =.4 ,color=colors )


    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor("#201D1D")

    # Add the probability percentages to the bars
    for i, v in enumerate([result['home_prob'], result['draw_prob'], result['away_prob']]):
        ax.text(v + 0.01, i, str(round(v*100, 2))+'%', color='#E1D3D3', fontweight='bold')

    # Set the plot title and axis labels
    # ax.set_title(f'{Fotmob_home_name} Vs {Fotmob_away_name} Outcome Probabilities',fontsize=12,
    #              color="#E1D3D3")

    ax.set_xlabel('Probability',fontsize=12,
                  color="#E1D3D3")

    # Create legend
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    ax.legend(handles, team_probs.keys(), loc='best',labelcolor='w',framealpha =.0)

    return ax


def simulate_matchshots_with_matrix(shot_df, max_goals=4):
    '''
    This function takes a match ID and simulates an outcome based on the shots
    taken by each team.
    '''

    shots = shot_df.copy()

    shots_home = shots[shots['Venue'] == 'Home']
    shots_away = shots[shots['Venue'] == 'Away']

    home_goals_probs = [poisson.pmf(i, sum(shots_home['xG'])) for i in range(max_goals+1)]
    away_goals_probs = [poisson.pmf(i, sum(shots_away['xG'])) for i in range(max_goals+1)]

    matrix = np.outer(home_goals_probs, away_goals_probs)

    return matrix


def plot_score_probability_matrix(ax, cmap, data):

    Fotmob_home_name = data[data['Venue'] == 'Home']
    Fotmob_away_name = data[data['Venue'] == 'Away']
    Fotmob_home_name = Fotmob_home_name['TeamName'].iloc[0]
    Fotmob_away_name = Fotmob_away_name['TeamName'].iloc[0]

    # Call the function to get the probability matrix
    matrix = simulate_matchshots_with_matrix(shot_df=data, max_goals=5)


    # Create the plot
    mask = matrix < 0.005
    sns.heatmap(matrix, cmap=cmap, ax=ax, square=True, annot=True, fmt='.1%',  linewidths=.25 ,linestyle= '--', linecolor='#201D1D', mask= mask,cbar=False)
    ax.set_ylim(0, matrix.shape[0])

    # ax.set_ylim(0, matrix['max_goals']+1)


    ax.set_xlabel(Fotmob_away_name, fontsize=8, color='#E1D3D3')
    ax.set_ylabel(Fotmob_home_name, fontsize=8, color='#E1D3D3')


    ax.tick_params(axis='x', colors='#E1D3D3')
    ax.tick_params(axis='y', colors='#E1D3D3')



    return ax




def plot_goals_prob(ax, data, team_id, color):

    data = data.copy()
    data = data[data['teamId'] == team_id]

    xG = data['xG'].tolist()

    max_goals = 6

    totalxg = np.cumsum(xG)[-1]

    gProbs = [poisson.pmf(i,totalxg) for i in range(max_goals)]

    plot_xg = np.round((np.array(gProbs) * 100),2)

    # First plot
    ax.bar(range(len(plot_xg)),plot_xg,color=color,edgecolor='#343a40', lw=1., zorder = 0)
    ax.set_xticks(range(len(plot_xg)))
    ax.axvline(totalxg,linestyle='--',color='#E1D3D3',linewidth=2.5 ,alpha=0.7)
    ax.text(totalxg,38,'xG',color='#E1D3D3', ha ='right', fontsize=8)
    ax.set_xlabel('Goals',color='#E1D3D3')

    ax.set_yticks([])
    ax.grid(False)
    for i in range(len(plot_xg)):
        ax.annotate(str(plot_xg[i])+"%", (range(len(plot_xg))[i], plot_xg[i]+1),c='#E1D3D3',size=8,ha='center',va='center',fontweight='bold')

    ax.set_yticks(np.linspace(0,40,5))
    ax.tick_params(axis='x', colors='#E1D3D3')
    ax.set_yticks([])
    ax.set_ylabel('')

    return ax


#%%

#%%
