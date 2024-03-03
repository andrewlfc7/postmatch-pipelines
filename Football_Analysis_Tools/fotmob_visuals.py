import json
from matplotlib.patheffects import withStroke

import matplotlib.pyplot as plt
import pandas as pd
import requests
from mplsoccer import Pitch, VerticalPitch
from unidecode import unidecode



def plot_match_shotmap(ax, match_id:int, homecolor, awaycolor):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')
    data = json.loads(response.content)

    matchId = data['general']['matchId']
    homeTeam = data['general']['homeTeam']
    awayTeam = data['general']['awayTeam']

    shot_data = data['content']['shotmap']['shots']
    df_shot = pd.DataFrame(shot_data)
    df_shot['matchId'] = matchId

    team_dict_name = {homeTeam['id']: homeTeam['name'], awayTeam['id']: awayTeam['name']}
    df_shot['teamName'] = df_shot['teamId'].map(team_dict_name)

    team_dict = {homeTeam['id']: 'Home', awayTeam['id']: 'Away'}
    df_shot['Venue'] = df_shot['teamId'].map(team_dict)


    # Set team color for both home and away team
    # home_color = h_data.teamColor.iloc[0]
    # away_color = a_data.teamColor.iloc[0]

    h_data = df_shot[df_shot['Venue'] == 'Home']
    a_data = df_shot[df_shot['Venue'] == 'Away']
    # Separate own goals for home and away teams
    Home_own_goals = a_data[(a_data['eventType'] == 'Goal') & (a_data['isOwnGoal'] == True)]
    Away_own_goals = h_data[(h_data['eventType'] == 'Goal') & (h_data['isOwnGoal'] == True)]

    # Home_goals = h_data[(h_data['eventType'] == 'Goal') & (h_data['isOwnGoal'] == False)]
    # Away_goals = a_data[(a_data['eventType'] == 'Goal') & (a_data['isOwnGoal'] == False)]
    Home_goals = pd.concat([h_data[(h_data['eventType'] == 'Goal') & (h_data['isOwnGoal'] == False)], Away_own_goals])
    Away_goals = pd.concat([a_data[(a_data['eventType'] == 'Goal') & (a_data['isOwnGoal'] == False)], Home_own_goals])

    Home_shots = h_data[h_data['eventType'] != 'Goal']
    Away_shots = a_data[a_data['eventType'] != 'Goal']



    pitch = Pitch(
        linewidth=2.5,
        pitch_color='#201D1D',
        pitch_type="uefa",
        half=False,
        goal_type='box',
        line_color='black'
    )
    pitch.draw(ax=ax)



    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0] - 0.5, pos_y[-1] + 0.5], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0] - 0.5, pos_x[-1] + 0.5], [y, y], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)


    ax.scatter(105 - Home_shots.x,
               70 - Home_shots.y,
               c=homecolor,
               s=Home_shots.expectedGoals.fillna(0) * 120,
               marker='o',
               alpha=0.8,
               label='Shots'
               )

    ax.scatter(105 - Home_goals.x,
               70 - Home_goals.y,
               c="#f28482",
               s=Home_goals.expectedGoals.fillna(0) * 120,
               marker='o',
               alpha=0.8,
               label='Goal'
               )

    ax.scatter(Away_shots.x,
               Away_shots.y,
               c=awaycolor,
               s=Away_shots.expectedGoals.fillna(0) * 120,
               alpha=0.8,
               label='Shot',
               marker='o'
               )

    ax.scatter(Away_goals.x,
               Away_goals.y,
               c="#f28482",
               s=Away_goals.expectedGoals.fillna(0) * 120,
               marker='o',
               alpha=0.8,
               )
    for eg in [.10,.25,.50]:
        ax.scatter([], [], c='k', alpha=0.3, s=eg * 90 ,
                   label=str(eg)+'xG')

    legend = ax.legend(scatterpoints=1, markerscale=.6, labelcolor='w',columnspacing=.02, labelspacing=.02, ncol=3,
                       loc='upper center', fontsize=6, framealpha= .00, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')


    return ax



def plot_match_xgflow(ax, match_id: int, homecolor,awaycolor):
    response = requests.get(
        f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')

    data = json.loads(response.content)
    homeTeam = data['general']['homeTeam']
    awayTeam = data['general']['awayTeam']

    shot_data = data['content']['shotmap']['shots']
    df_shot = pd.DataFrame(shot_data)
    df_shot['min'] = df_shot['min'].astype(int)
    df_shot['xG'] = df_shot['expectedGoals'].astype(float)

    df_shot['min'] = df_shot['min'].astype(int)

    xg_flow = df_shot[
        ['teamId', 'situation', 'eventType', 'expectedGoals', 'playerName', 'min', 'teamColor', 'isOwnGoal']]


    team_dict_name = {homeTeam['id']: homeTeam['name'], awayTeam['id']: awayTeam['name']}
    xg_flow.loc[:, 'teamName'] = xg_flow['teamId'].map(team_dict_name).copy()


    # Create a dictionary mapping team IDs to team names
    team_dict = {homeTeam['id']: 'Home', awayTeam['id']: 'Away'}

    xg_flow.loc[:, 'Venue'] = xg_flow['teamId'].map(team_dict).copy()
    a_xG = [0]
    h_xG = [0]
    a_min = [0]
    h_min = [0]

    hteam = xg_flow[xg_flow['Venue'] == 'Home']
    ateam = xg_flow[xg_flow['Venue'] == 'Away']

    a_xG.extend(ateam['expectedGoals'])
    a_min.extend(ateam['min'])

    h_xG.extend(hteam['expectedGoals'])
    h_min.extend(hteam['min'])

    def nums_cumulative_sum(nums_list):
        return [sum(nums_list[:i+1]) for i in range(len(nums_list))]

    a_cumulative = nums_cumulative_sum(a_xG)
    h_cumulative = nums_cumulative_sum(h_xG)

    #this is used to find the total xG. It just creates a new variable from the last item in the cumulative list
    alast = round(a_cumulative[-1],2)
    hlast = round(h_cumulative[-1],2)


    # append 90 to a_min and h_min and the last cum_xg value to a_cumulative and h_cumulative
    a_min.append(90)
    h_min.append(90)
    a_cumulative.append(alast)
    h_cumulative.append(hlast)


    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Remove ticks and labels for the X and Y axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')


    ytick = max(h_cumulative[-1],a_cumulative[-1])
    if ytick > 3:
        space = 0.5
    else:
        space = 0.25

    # home_color = hteam.teamColor.iloc[0]
    # away_color = ateam.teamColor.iloc[0]


    hteam.loc[:, 'cum_xg'] = h_cumulative[:len(hteam.index)].copy()
    ateam.loc[:, 'cum_xg'] = a_cumulative[:len(ateam.index)].copy()

    x1 = hteam[hteam['eventType']=='Goal']['min'].tolist()
    y1 =hteam[hteam['eventType']=='Goal']['cum_xg'].tolist()

    x2 = ateam[ateam['eventType']=='Goal']['min'].tolist()
    y2 =ateam[ateam['eventType']=='Goal']['cum_xg'].tolist()



    scatter = ax.scatter(x=x1, y=y1, zorder=3, alpha=.6, c=homecolor, edgecolors= '#343a40' , s=300)
    scatter.set_path_effects([withStroke(linewidth=8, foreground=homecolor,alpha = .2)])

    for x, y in zip(x1, y1):
        text_list = ['Goal']
        for txt in text_list:
            ax.text(x, y, txt, ha='center', va='center', color='white', fontweight='bold', fontsize=4)


    scatter2 = ax.scatter(x=x2, y=y2, zorder=3, alpha=.6, c=awaycolor, edgecolors= '#343a40' , s=300)
    scatter2.set_path_effects([withStroke(linewidth=8, foreground=awaycolor, alpha = .2)])

    for x, y in zip(x2, y2):
        text_list = ['Goal']
        for txt in text_list:
            ax.text(x, y, txt, ha='center', va='center', color='white', fontweight='bold', fontsize=4)


    #plot the step graphs
    ax.step(x=a_min,y=a_cumulative,color=awaycolor,label=ateam['teamName'].iloc[0],linewidth=2, linestyle='--', where='post')
    ax.step(x=h_min,y=h_cumulative,color=homecolor,label=hteam['teamName'].iloc[0],linewidth=2, linestyle='solid',where='post')

    plt.fill_between(a_min,a_cumulative, step='post',interpolate=True, alpha=0.5, color=awaycolor)
    plt.fill_between(h_min,h_cumulative,step= 'post' , interpolate=True,alpha=0.5, color=homecolor)

    ax.legend(facecolor='#D1D1D1', labelcolor='w', framealpha=.0)

    plt.xticks([], [])
    plt.yticks([], [])


    return ax



def plot_player_shotmap(ax, match_id: int, player_name):
    player_name = unidecode(player_name)
    response = requests.get(
        f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')
    data = json.loads(response.content)

    shot_data = data['content']['shotmap']['shots']
    df_shot = pd.DataFrame(shot_data)

    df_shot = df_shot[df_shot['playerName'] == player_name]
    shots = df_shot[df_shot['eventType'] != 'Goal']

    df_goal = df_shot[df_shot['eventType'] == 'Goal']

    pitch = pitch = VerticalPitch(
        pitch_color='#201D1D',
        half=True,
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='uefa',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    # Remember that we need to invert the axis!!
    for x in pos_x[1:-1]:
        ax.plot([pos_y[0], pos_y[-1]], [x, x], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([y, y], [pos_x[0], pos_x[-1]], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)

    ax.scatter(shots.y,
               shots.x,
               c='#7371FC',
               s=shots.expectedGoals * 120,
               alpha=.8,
               label = 'Shot'
               )

    ax.scatter(df_goal.y,
               df_goal.x,
               c="#5EB39E",
               s=df_goal.expectedGoals * 120,
               marker='o',
               alpha=.8,
               label = 'Goal'
               )

    for eg in [.10,.25,.50]:
        ax.scatter([], [], c='k', alpha=0.3, s=eg * 90 * 6,
                   label=str(eg)+'xG')

    legend = ax.legend(scatterpoints=1, markerscale=.4, columnspacing=.02, labelcolor='white',labelspacing=.02, ncol=6,
                       loc='upper center', fontsize=6, framealpha=.0,bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')


    return ax



#%%

#%%
