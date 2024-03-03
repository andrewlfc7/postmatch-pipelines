import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import to_rgba
from mplsoccer import Pitch, VerticalPitch
from highlight_text import fig_text
import matplotlib.patches as mpatches
import scipy.ndimage as ndi

from scipy.stats import zscore
import scipy

from sklearn.cluster import KMeans
from scipy.interpolate import make_interp_spline


def compute_contested_zones(matchID, team_name, data):
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.1,
        line_color='black',
        pad_top=10,
        corner_arcs=True
    )
    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y
    data = data.copy()
    df_match = data[data['match_id'] == matchID]
    # -- Adjust opposition figures
    df_match.loc[:, 'x'] = [100 - x if y != team_name else x for x, y in zip(df_match['x'], df_match['team_name'])]
    df_match.loc[:, 'y'] = [100 - x if y != team_name else x for x, y in zip(df_match['y'], df_match['team_name'])]
    df_match = df_match.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    df_match = df_match.assign(bins_y=lambda x: pd.cut(x.y, bins=list(pos_y) + [105]))
    df_match_groupped = df_match.groupby(['bins_x', 'bins_y', 'team_name', 'match_id'])['isTouch'].sum().reset_index(
        name='touches')
    df_team = df_match_groupped[df_match_groupped['team_name'] == team_name]
    df_oppo = df_match_groupped[df_match_groupped['team_name'] != team_name].rename(
        columns={'team_name': 'opp_name', 'touches': 'opp_touches'})
    df_plot = pd.merge(df_team, df_oppo, on=['bins_x', 'bins_y'])
    df_plot = df_plot.assign(ratio=lambda x: x.touches / (x.touches + x.opp_touches))
    df_plot['left_x'] = df_plot['bins_x'].apply(lambda x: x.left).astype(float)
    df_plot['right_x'] = df_plot['bins_x'].apply(lambda x: x.right).astype(float)
    df_plot['left_y'] = df_plot['bins_y'].apply(lambda x: x.left).astype(float)
    df_plot['right_y'] = df_plot['bins_y'].apply(lambda x: x.right).astype(float)
    return df_plot


def plot_zone_dominance(ax, match_id, team_name, homecolor, awaycolor, zonecolor, data):
    data_plot = data.copy()
    data_plot = compute_contested_zones(match_id, team_name, data=data_plot)
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        pad_top=10,
        corner_arcs=True
    )
    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
                condition_bounds = (data_plot['left_x'] >= lower_x) & (data_plot['right_x'] <= upper_x) & (
                        data_plot['left_y'] >= lower_y) & (data_plot['right_y'] <= upper_y)
                data_point = data_plot[condition_bounds]['ratio'].iloc[0]
                if data_point > .55:
                    color = homecolor
                elif data_point < .45:
                    color = awaycolor
                else:
                    color = zonecolor
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=0.75,
                    ec='None'
                )
            except:
                continue

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    home_team = data[data['Venue']=="Home"]
    home_team_name = home_team['team_name'].iloc[0]

    away_team = data[data['Venue']=="Away"]
    away_team_name = away_team['team_name'].iloc[0]

    bbox_pad = .6
    bboxprops = {'linewidth': 0, 'pad': bbox_pad}
    ax.legend(handles=[mpatches.Patch(facecolor=homecolor, edgecolor='k', label= 'Areas Where ' + home_team_name + '\nHad Majority Of The Touches'),mpatches.Patch(facecolor=awaycolor, edgecolor='k', label= 'Areas Where ' + away_team_name + '\nHad Majority Of The Touches'),mpatches.Patch(facecolor=zonecolor, edgecolor='k', label='Contested \nAreas')],
              bbox_to_anchor=(0.5, -0.10),
              loc='lower center',
              fontsize=6,
              labelspacing =.25,
              markerscale= .6,
              labelcolor ='w',
              columnspacing=.5,
              framealpha=.0,
              ncol=3)

    return ax





def plot_team_offensive_actions(ax, match_id, team_id, data,color):
    data = data[(data['is_open_play'] == True) & (data['event_type'] != 'Pass')]

    data_offensive = data.copy()

    data_offensive = data_offensive[(data_offensive['match_id'] == match_id) & (data_offensive['teamId'] == team_id)]

    data_offensive = data_offensive[data_offensive['x'] >= 50].reset_index(drop=True)

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=True
    )

    pitch.draw(ax=ax)


    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_offensive = data_offensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_offensive = data_offensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_offensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * .8,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.2,
                    ec='None'
                )
            counter += 1

    unique_actions = data_offensive['event_type'].unique()
    actions = list(unique_actions)
    markers = ['o', 'X', 'v', 's', '^']
    for a, m in zip(actions, markers):
        if a == 'TakeOn':
            a_label = 'Take On'
        else:
            a_label = a
        num_data_points = len(data_offensive[data_offensive['event_type'] == a])
        marker_size = 40 + num_data_points * 0.5
        ax.scatter(data_offensive[data_offensive['event_type'] == a].x,
                   data_offensive[data_offensive['event_type'] == a].y, s=marker_size, alpha=0.85, lw=0.85,
                   fc='#3c6e71', ec='#2F2B2B', zorder=3, marker=m, label=a_label)

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6,labelcolor='w', framealpha=.0, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor("#201D1D")
    return ax


def plot_team_defensive_actions_opp_half(ax, match_id: int, team_id: int, data, color):
    data =data.copy()
    data = data[data['is_open_play'] == True]
    data = data[(data['is_open_play'] == True) & (data['event_type'] != 'Pass')]

    data_recoveries = data[data['won_possession'] == True]

    data_recoveries = data_recoveries[
        (data_recoveries['match_id'] == match_id) & (data_recoveries['teamId'] == team_id)]

    data_defensive = data_recoveries.copy()

    data_defensive = data_defensive[data_defensive['x'] >= 50].reset_index(drop=True)

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=True
    )

    pitch.draw(ax=ax)


    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_defensive = data_defensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_defensive = data_defensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_defensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * .8,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.2,
                    ec='None'
                )
            counter += 1

    unique_actions = data_defensive['event_type'].unique()
    actions = list(unique_actions)
    markers = ['o', 'X', 'v', 's', '^']
    for a, m in zip(actions, markers):
        if a == 'BallRecovery':
            a_label = 'Ball recovery'
        else:
            a_label = a
        num_data_points = len(data_defensive[data_defensive['event_type'] == a])
        marker_size = 40 + num_data_points * 0.5
        ax.scatter(data_defensive[data_defensive['event_type'] == a].x,
                   data_defensive[data_defensive['event_type'] == a].y, s=marker_size, alpha=0.85, lw=0.85,
                   fc='#3c6e71', ec='#2F2B2B', zorder=3, marker=m, label=a_label)

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, labelcolor='w', framealpha=.0,bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor("#201D1D")
    return ax




def plot_player_hull_team(player_df, ax, poly_edgecolor='#A0A0A0', poly_facecolor='#379A95', poly_alpha=0.3,
                          scatter_edgecolor='#0C0F0A', scatter_facecolor='#3c6e71',avg_marker_size=600, sd=int):

    x_mean = player_df['x'].mean()
    y_mean = player_df['y'].mean()
    x_std = player_df['x'].std()
    y_std = player_df['y'].std()

    new_player_df = player_df[(np.abs(player_df['x'] - x_mean) < sd * x_std) & (np.abs(player_df['y'] - y_mean) < sd * y_std)]

    pitch = Pitch()
    # Draw convex hull polygon
    hull = pitch.convexhull(new_player_df.x, new_player_df.y)
    poly = pitch.polygon(hull, ax=ax, edgecolor=poly_edgecolor, facecolor=poly_facecolor, alpha=poly_alpha)

    # Draw scatter plot
    scatter = pitch.scatter(new_player_df.x, new_player_df.y, ax=ax, edgecolor=scatter_edgecolor, facecolor=scatter_facecolor,alpha=.30)

    # Draw average location marker
    pitch.scatter(new_player_df.x_avg, new_player_df.y_avg, ax=ax, edgecolor=scatter_edgecolor, facecolor=poly_facecolor,
                  s=avg_marker_size, marker='o', alpha=.20)

    # Add player initials as labels
    for i, row in new_player_df.iterrows():
        ax.text(row['x_avg'], row['y_avg'], row['initials'], fontsize=4, color='k', ha='center', va='center')

    return ax




# %%

# %%

def plot_player_passmap_opta(ax, player_name, data):
    data = data.copy()

    # data = data[(data['is_open_play']) & (~data['assist']) & (~data['key_pass'])]



    data = data[data['playerName'] == player_name]
    data = data[data['event_type'] == 'Pass']
    data = data[data['outcomeType']=='Successful']
    # fig = plt.figure(figsize = (4,4), dpi = 900)
    # ax = plt.subplot(111)
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)


    pitch.arrows(data.x, data.y, data.endX, data.endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#264653', alpha=.40, label = 'Pass',zorder=1, ax=ax)

    pitch.arrows(data[data['key_pass'] == True]
                 .x, data[data['key_pass'] == True]
                 .y, data[data['key_pass'] == True]
                 .endX, data[data['key_pass'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#a28497', label = 'Key Pass',alpha=.8, zorder=3, ax=ax)

    pitch.arrows(data[data['assist'] == True]
                 .x, data[data['assist'] == True]
                 .y, data[data['assist'] == True]
                 .endX, data[data['assist'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#e59878', alpha=.8, zorder=5,label ='Assist' , ax=ax)

    pitch.arrows(data[data['pre_assist']==True]
                 .x,data[data['pre_assist']==True]
                 .y,data[data['pre_assist']==True]
                 .endX, data[data['pre_assist']==True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#d1495b',label ='Pre Assist' ,alpha=.8,zorder=4, ax=ax)

    pitch.arrows(data[data['FinalThirdPasses'] == True]
                 .x, data[data['FinalThirdPasses'] == True]
                 .y, data[data['FinalThirdPasses'] == True]
                 .endX, data[data['FinalThirdPasses'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#28aca6', alpha=.8, zorder=2,label = 'Final 1/3 Pass', ax=ax)


    pitch.arrows(data[data['is_progressive'] == True]
                 .x, data[data['is_progressive'] == True]
                 .y, data[data['is_progressive'] == True]
                 .endX, data[data['is_progressive'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#65a46d', label = 'Progressive Pass',alpha=.8, zorder=2, ax=ax)


    pitch.arrows(data[data['switch'] == True]
                 .x, data[data['switch'] == True]
                 .y, data[data['switch'] == True]
                 .endX, data[data['switch'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#edae49', label = 'Switch/Long Pass',alpha=.8, zorder=3, ax=ax)

    pitch.arrows(data[data['is_cross'] == True]
                 .x, data[data['is_cross'] == True]
                 .y, data[data['is_cross'] == True]
                 .endX, data[data['is_cross'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#9381ff', label = 'Cross',alpha=.8, zorder=4, ax=ax)




# Add legend
    legend = ax.legend(ncol=4, loc='upper center', fontsize=6, markerscale=.0002,labelcolor='w',labelspacing=1,framealpha=.0,bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    return ax


# %%

def plot_convex_hull_opta_player(ax, player_name, data):
    data = data.copy()
    data = data[(data['playerName'] == player_name) & (data['is_open_play'] == True)]
    data = data[(data['isTouch']==True) & (data['event_type']!='Carry')]

    player_avg_pos = data.groupby('playerName')[['x', 'y']].mean()
    player_avg_pos['initials'] = player_avg_pos.index.str.split().str[0].str[0] + \
                                 player_avg_pos.index.str.split().str[-1].str[0]

    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    hull = pitch.convexhull(data.x, data.y)
    poly = pitch.polygon(hull, ax=ax, edgecolor='#2a9d8f', facecolor='#2a9d8f', alpha=0.3)

    ax.scatter(data.x, data.y, s=20, alpha=.6, edgecolors='#1A1D1A', color='#2a9d8f', zorder=1, label = 'Open Play Touch')

    pitch.scatter(player_avg_pos.x, player_avg_pos.y, ax=ax, edgecolor='#343a40', facecolor='#2a9d8f', s=400,
                  marker='o', alpha=.30, zorder=2,label = 'Average Postion')

    for i, row in player_avg_pos.iterrows():
        ax.text(row['x'], row['y'], row['initials'], fontsize=6, color='w', ha='center', va='center')
    legend = ax.legend(ncol=3, loc='upper center', fontsize=6, handlelength=2.5,markerscale=.4 ,framealpha=.0,labelcolor='w',handleheight=2.5, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


    return ax


# %%
def plot_players_defensive_actions_opta(ax, playerName, data,color):
    data = data.copy()
    data = data[data['is_open_play'] == True]
    # data = def_action(data)

    data = data[(data['playerName'] == playerName) & (data['outcomeType'] == 'Successful')]

    data_defensive = data.copy()

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_defensive = data_defensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_defensive = data_defensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_defensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

    unique_actions = data_defensive['event_type'].unique()
    actions = list(unique_actions)
    markers = ['o', 'X', 'v', 's', '^']
    for a, m in zip(actions, markers):
        if a == 'BallRecovery':
            a_label = 'Ball recovery'
        else:
            a_label = a
        num_data_points = len(data_defensive[data_defensive['event_type'] == a])
        marker_size = 40 + num_data_points * 0.5
        ax.scatter(data_defensive[data_defensive['event_type'] == a].x,
                   data_defensive[data_defensive['event_type'] == a].y, s=marker_size, alpha=0.85, lw=0.85,
                   fc='#3c6e71', ec='#2F2B2B', zorder=3, marker=m, label=a_label)

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6,framealpha=.0,labelcolor='w', bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax


def plot_players_offensive_actions_opta(ax, playerName, data,color):
    data = data.copy()
    data = data[data['is_open_play'] == True]
    data = data[(data['playerName'] == playerName) & (data['outcomeType'] == 'Successful')]

    data_offensive = data.copy()

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_offensive = data_offensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_offensive = data_offensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_offensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

    unique_actions = data_offensive['event_type'].unique()
    actions = list(unique_actions)
    markers = ['o', 'X', 'v', 's', '^']
    for a, m in zip(actions, markers):
        if a == 'BallRecovery':
            a_label = 'Ball recovery'
        else:
            a_label = a
        num_data_points = len(data_offensive[data_offensive['event_type'] == a])
        marker_size = 40 + num_data_points * 0.5
        ax.scatter(data_offensive[data_offensive['event_type'] == a].x,
                   data_offensive[data_offensive['event_type'] == a].y, s=marker_size, alpha=0.85, lw=0.85,
                   fc='#3c6e71', ec='#2F2B2B', zorder=3, marker=m, label=a_label)

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, framealpha=.0,labelcolor='w',bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    # ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax


def plot_carry_opta(ax, player_name, data):
    data = data.copy()
    data = data[data['playerName'] == player_name]
    data = data[data['event_type'] == 'Carry']
    # fig = plt.figure(figsize = (4,4), dpi = 900)
    # ax = plt.subplot(111)
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    pitch.arrows(data.x, data.y, data.endX, data.endY, color="#2a9d8f", width=1.5, headwidth=3, headlength=4, ax=ax)

    return ax



def plot_carry_player_opta(ax, player_name, data):
    data = data.copy()
    data = data[data['is_open_play'] == True]
    data = data[data['playerName'] == player_name]
    data = data[data['event_type'] == 'Carry']
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',
    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    pitch.arrows(data.x, data.y, data.endX, data.endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#264653', alpha=.40, label='Carry', zorder=1, ax=ax)

    pitch.arrows(data[data['progressive_carry'] == True]
                 .x, data[data['progressive_carry'] == True]
                 .y, data[data['progressive_carry'] == True]
                 .endX, data[data['progressive_carry'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#a28497', label='Progressive Carry', alpha=.8, zorder=3, ax=ax)

    pitch.arrows(data[data['is_carry_into_box'] == True]
                 .x, data[data['is_carry_into_box'] == True]
                 .y, data[data['is_carry_into_box'] == True]
                 .endX, data[data['is_carry_into_box'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#e59878', alpha=.8, zorder=5, label='Carry Into Box', ax=ax)

    # Add legend
    legend = ax.legend(ncol=4, loc='upper center', fontsize=6, markerscale=.0002, labelcolor='w', framealpha=.0,
                       bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    return ax




def plot_player_passes_rec_opta(ax, player_name, data):
    data = data.copy()
    # data = data[(data['is_open_play']) & (~data['assist']) & (~data['key_pass'])]

    data = data[data['pass_recipient'] == player_name]
    data = data[data['outcomeType']=='Successful']
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)


    pitch.arrows(data.x, data.y, data.endX, data.endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#264653', alpha=.40, label = 'Pass',zorder=1, ax=ax)

    pitch.arrows(data[data['key_pass'] == True]
                 .x, data[data['key_pass'] == True]
                 .y, data[data['key_pass'] == True]
                 .endX, data[data['key_pass'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#a28497', label = 'Key Pass',alpha=.8, zorder=3, ax=ax)

    pitch.arrows(data[data['assist'] == True]
                 .x, data[data['assist'] == True]
                 .y, data[data['assist'] == True]
                 .endX, data[data['assist'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#e59878', alpha=.8, zorder=5,label ='Assist' , ax=ax)

    pitch.arrows(data[data['pre_assist']==True]
                 .x,data[data['pre_assist']==True]
                 .y,data[data['pre_assist']==True]
                 .endX, data[data['pre_assist']==True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#d1495b',label ='Pre Assist' ,alpha=.8,zorder=4, ax=ax)

    pitch.arrows(data[data['FinalThirdPasses'] == True]
                 .x, data[data['FinalThirdPasses'] == True]
                 .y, data[data['FinalThirdPasses'] == True]
                 .endX, data[data['FinalThirdPasses'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#28aca6', alpha=.8, zorder=2,label = 'Final 1/3 Pass', ax=ax)


    pitch.arrows(data[data['is_progressive'] == True]
                 .x, data[data['is_progressive'] == True]
                 .y, data[data['is_progressive'] == True]
                 .endX, data[data['is_progressive'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#65a46d', label = 'Progressive Pass',alpha=.8, zorder=2, ax=ax)


    pitch.arrows(data[data['switch'] == True]
                 .x, data[data['switch'] == True]
                 .y, data[data['switch'] == True]
                 .endX, data[data['switch'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#edae49', label = 'Switch/Long Pass',alpha=.8, zorder=3, ax=ax)

    pitch.arrows(data[data['is_cross'] == True]
                 .x, data[data['is_cross'] == True]
                 .y, data[data['is_cross'] == True]
                 .endX, data[data['is_cross'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#9381ff', label = 'Cross',alpha=.8, zorder=4, ax=ax)


    # Add legend
    legend = ax.legend(ncol=4, loc='upper center', fontsize=6, markerscale=.0002,labelcolor='w',labelspacing=1,framealpha=.0,bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')


    return ax


def plot_carry_into_box_team_opta(ax, team_id, data):
    data = data.copy()
    data = data[data['teamId'] == team_id]
    data = data[data['event_type'] == 'Carry']
    # fig = plt.figure(figsize = (4,4), dpi = 900)
    # ax = plt.subplot(111)
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',

    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    # pitch.arrows(data.x, data.y, data.endX, data.endY, color="#2a9d8f", width=1.5, headwidth=3, headlength=4, ax=ax)

    # Plot arrows for carries into the box in blue
    pitch.arrows(data[data['is_carry_into_box'] == True].x, data[data['is_carry_into_box'] == True].y,
                 data[data['is_carry_into_box'] == True].endX, data[data['is_carry_into_box'] == True].endY,
                 color='#4A7B9D', width=1.5, headwidth=3, headlength=4, alpha=.7, ax=ax, label='Carries into box')

    # Plot arrows for carries outside the box in green
    pitch.arrows(data[data['is_carry_into_box'] != True].x, data[data['is_carry_into_box'] != True].y,
                 data[data['is_carry_into_box'] != True].endX, data[data['is_carry_into_box'] != True].endY,
                 color='#2a9d8f', width=1.5, headwidth=3, headlength=4, alpha=.7, ax=ax, label='Carries')

    # Add legend
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    return ax


def plot_carry_into_box_player_opta(ax, player_name, data):
    data = data.copy()
    data = data[data['playerName'] == player_name]
    data = data[data['event_type'] == 'Carry']
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',
    )
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    # Plot arrows for carries into the box in blue
    pitch.arrows(data[data['is_carry_into_box'] == True].x, data[data['is_carry_into_box'] == True].y,
                 data[data['is_carry_into_box'] == True].endX, data[data['is_carry_into_box'] == True].endY,
                 color='#4A7B9D', width=1.5, headwidth=3, headlength=4, alpha=.7, ax=ax, label='Carries into box')

    # Plot arrows for carries outside the box in green
    pitch.arrows(data[data['is_carry_into_box'] != True].x, data[data['is_carry_into_box'] != True].y,
                 data[data['is_carry_into_box'] != True].endX, data[data['is_carry_into_box'] != True].endY,
                 color='#2a9d8f', width=1.5, headwidth=3, headlength=4, alpha=.7, ax=ax, label='Carries')

    # Add legend
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')



    return ax


def plot_team_touch_heatmap_opta(ax, match_id, team_id, data):
    data = data[data['is_open_play'] == True]


    data_offensive = data.copy()

    data_offensive = data_offensive[(data_offensive['match_id'] == match_id) & (data_offensive['teamId'] == team_id)]

    # data_offensive = data_offensive[(data_offensive['event_type'] == 'Pass') & (data_offensive['isTouch'] == true)]

    data_offensive = data_offensive[data_offensive['x'] > 50].reset_index(drop=True)

    data_offensive = data_offensive[(data_offensive['event_type'] == 'Pass') & (data_offensive['isTouch'] == True)]

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=True
    )

    pitch.draw(ax=ax)

    #fig = plt.figure(figsize=(8, 4))

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_offensive = data_offensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_offensive = data_offensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_offensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='#4A9FA2',
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

        #ax.scatter(data_offensive.x, data_offensive.y, s=20, alpha=0.6, lw=0.85,
        #          fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label = "Touch")

        #pitch.arrows(data_offensive.x, data_offensive.y, data_offensive.endX, data_offensive.endY, color="#2a9d8f", width=1.5, headwidth=3, headlength=4, ax=ax)


    ax.scatter(data_offensive.x, data_offensive.y, s=20, alpha=0.6, lw=0.85,
               fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label="Open Play Touches")

    legend = ax.legend(loc='upper center', fontsize=6,labelcolor='w', framealpha=.0, bbox_to_anchor=(.5, 0.03))


    #legend = ax.legend(['Open Play Touch'], loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))

    #legend = ax.legend( loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax


def plot_team_touch_heatmap_full_opta(ax, match_id, team_id, data):
    data = data[data['is_open_play'] == True]


    data_touch = data.copy()

    data_touch = data_touch[(data_touch['match_id'] == match_id) & (data_touch['teamId'] == team_id)]

    # data_touch = data_touch[(data_touch['event_type'] == 'Pass') & (data_touch['isTouch'] == true)]

    #data_touch = data_touch[data_touch['x'] > 50].reset_index(drop=True)

    data_touch = data_touch[(data_touch['event_type'] == 'Pass') & (data_touch['isTouch'] == True)]

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    #fig = plt.figure(figsize=(8, 4))

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_touch = data_touch.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_touch = data_touch.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_touch.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='#4A9FA2',
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1

        #ax.scatter(data_offensive.x, data_offensive.y, s=20, alpha=0.6, lw=0.85,
        #          fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label = "Touch")

        #pitch.arrows(data_offensive.x, data_offensive.y, data_offensive.endX, data_offensive.endY, color="#2a9d8f", width=1.5, headwidth=3, headlength=4, ax=ax)


    ax.scatter(data_touch.x, data_touch.y, s=20, alpha=0.6, lw=0.85,
               fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label="Open Play Touches")

    legend = ax.legend( loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))


    #legend = ax.legend(['Open Play Touch'], loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))

    #legend = ax.legend( loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax




def plot_xT_flow_chart(ax, data,homecolor, awaycolor):
    home_team = data[data['Venue'] == 'Home']['team_name'].iloc[0]
    away_team = data[data['Venue'] == 'Away']['team_name'].iloc[0]

    # Further manipulation based on team
    data_list = []
    for name in [home_team, away_team]:
        if name == home_team:
            data_list.append(
                data.loc[(data['team_name'] == name) & (data['Venue'] == 'Home')]
                .assign(xThreat_gen=lambda x: np.where(x['xThreat_gen'] > 0, x['xThreat_gen'], -x['xThreat_gen']))
            )
        else:
            data_list.append(
                data.loc[(data['team_name'] == name) & (data['Venue'] == 'Away')]
                .assign(xThreat_gen=lambda x: np.where(x['xThreat_gen'] > 0, -x['xThreat_gen'], x['xThreat_gen']))
            )

    # Combine datasets and group by minute
    df = (pd.concat(data_list)
          .groupby('minute')
          .agg({'xThreat_gen': 'sum'})
          .reset_index()
          )

    # Interpolate data for a smoother line which goes through all plotted points
    spline = make_interp_spline(df['minute'], df['xThreat_gen'], k=3)
    spline_int = pd.DataFrame({'minute': np.linspace(df['minute'].min(), df['minute'].max(), 2000)})
    spline_int['xThreat_gen'] = spline(spline_int['minute'])
    spline_int['team_name'] = np.where(spline_int['xThreat_gen'] > 0, home_team, away_team)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels([])
    ax.set_facecolor("#201D1D")
    ax.grid(False)

    # Plot and fill between lines
    for name, color, hatch in zip([home_team, away_team], [homecolor, awaycolor], ['////', '\\\\\\\\']):
        ax.plot(spline_int['minute'], np.where(spline_int['team_name'] == name, spline_int['xThreat_gen'], np.nan), color=color, lw=1)
        ax.fill_between(spline_int['minute'], spline_int['xThreat_gen'], where=spline_int['team_name'] == name, color=color, alpha=0.4, interpolate=True, edgecolor=color, hatch=hatch, lw=0)

    return ax



def plot_team_touch_heatmap_opta_halfpitch(ax, match_id, team_id, data,color):
    data = data[data['is_open_play'] == True]


    data_offensive = data.copy()

    data_offensive = data_offensive[(data_offensive['match_id'] == match_id) & (data_offensive['teamId'] == team_id)]

    # data_offensive = data_offensive[(data_offensive['event_type'] == 'Pass') & (data_offensive['isTouch'] == true)]

    data_offensive = data_offensive[data_offensive['x'] > 50].reset_index(drop=True)

    data_offensive = data_offensive[(data_offensive['event_type'] == 'Pass') & (data_offensive['isTouch'] == True)]

    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=True
    )

    pitch.draw(ax=ax)

    #fig = plt.figure(figsize=(8, 4))

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_offensive = data_offensive.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_offensive = data_offensive.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_offensive.groupby(['bins_x', 'bins_y']).size().reset_index(name='count')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['count'].iloc[0] / data_grouped['count'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * .8,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.2,
                    ec='None'
                )
            counter += 1



    ax.scatter(data_offensive.x, data_offensive.y, s=20, alpha=0.6, lw=0.85,
               fc='#3c6e71', ec='#2F2B2B', zorder=3, marker='o', label="Open Play Touches")

    legend = ax.legend( loc='upper center', fontsize=6,framealpha=.0, labelcolor='w' , bbox_to_anchor=(.5, 0.03))


    #legend = ax.legend(['Open Play Touch'], loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))

    #legend = ax.legend( loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax



def plot_passes_into_box_team_opta(ax, teamid, data):
    data = data.copy()
    data = data[data['teamId'] == teamid]
    data = data[data['is_open_play']==True]
    data = data[(data['event_type'] == 'Pass') | (data['outcomeType'] == 'Successful')]


    pitch = VerticalPitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.25,
        line_color='black',
        half=True
    )
    pitch.draw(ax=ax)


    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    # Remember that we need to invert the axis!!
    for x in pos_x[1:-1]:
        ax.plot([pos_y[0], pos_y[-1]], [x,x], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([y,y], [pos_x[0], pos_x[-1]], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)



    # Plot arrows for passes into the box in blue
    pitch.arrows(data[data['is_pass_into_box'] == True].x, data[data['is_pass_into_box'] == True].y,
                 data[data['is_pass_into_box'] == True].endX, data[data['is_pass_into_box'] == True].endY,
                 color = '#2a9d8f', width=1.5, headwidth=3, headlength=4, alpha= .8, ax=ax, label='Passes into box')




    # Add legend
    legend = ax.legend(ncol=5, loc='upper center', fontsize=6, bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    return ax



def plot_pass_map_with_xT_away(ax, teamId:int, data,cmap_name='coolwarm'):
    """Parameters:
    ax (Matplotlib Axes object): Axes object to plot the map on
    teamId (int): The Opta ID of the team to plot the map for
    data (dict): A dictionary containing Opta event data"""

    data = data.copy()
    data =data[data['teamId'] == teamId]

    cmap = plt.get_cmap(cmap_name)

    def get_passes_df(events_dict):
        df = pd.DataFrame(events_dict)
        # create receiver column based on the next event
        # this will be correct only for successfull passes
        df["pass_recipient"] = df["playerName"].shift(-1)
        # filter only passes

        passes_ids = df.index[df['event_type'] == 'Pass']
        df_passes = df.loc[
            passes_ids, ["id", "minute", "x", "y", "endX", "endY", "teamId", "playerId", "playerName", "event_type",
                         "outcomeType", "pass_recipient",'isTouch','xThreat_gen']]

        return df_passes


    passes_df = get_passes_df(data)
    passes_df
    passes_df = passes_df[passes_df['outcomeType'] == 'Successful']
    #find the first subsititution and filter the successful dataframe to be less than that minute
    subs = data[data['event_type'] == 'SubstitutionOff']
    subs
    subs = subs['minute']
    firstSub = subs.min()

    passes_df = passes_df[passes_df['minute'] < firstSub]

    average_locs_and_count_df = (passes_df.groupby('playerName')
                                 .agg({'x': ['mean'], 'y': ['mean', 'count']}))
    average_locs_and_count_df.columns = ['x', 'y', 'count']
    average_locs_and_count_df
    #average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position']],
    #                                                            on='playerId', how='left')
    #average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')





    pass_between = passes_df.groupby(['playerName', 'pass_recipient']).agg(total_xt=('xThreat_gen', 'sum'),
                                                                           pass_count=('id', 'count')).reset_index()

    #pass_between = passes_df.groupby(['playerName', 'pass_recipient'])['xT'].sum().reset_index()
    #pass_between.rename({'xT': 'total_xt', 'id': 'pass_count'}, axis='columns', inplace=True)




    pass_between = pass_between.merge(average_locs_and_count_df, left_on='playerName', right_index=True)
    pass_between = pass_between.merge(average_locs_and_count_df, left_on='pass_recipient', right_index=True,
                                      suffixes=['', '_end'])


    pass_between = pass_between[pass_between['pass_count'] >= 3]

    # Group passes by playerId and sum their xt values
    player_xt_df = passes_df.groupby('playerName')['xThreat_gen'].sum().reset_index()

    # Merge with the original player locations dataframe
    average_locs_and_count_df = pd.merge(average_locs_and_count_df, player_xt_df, on='playerName')

    pitch = Pitch(pitch_type='opta')


    pass_between['x'] = pitch.dim.right - pass_between['x']
    pass_between['y'] = pitch.dim.right - pass_between['y']
    pass_between['x_end'] = pitch.dim.right - pass_between['x_end']
    pass_between['y_end'] = pitch.dim.right - pass_between['y_end']
    average_locs_and_count_df['x'] = pitch.dim.right - average_locs_and_count_df['x']
    average_locs_and_count_df['y'] = pitch.dim.right - average_locs_and_count_df['y']



    MAX_MARKER_SIZE = 1200
    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']
                                                / average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)

    MAX_LINE_WIDTH = 16
    pass_between['width'] = (pass_between.pass_count / pass_between.pass_count.max() *
                             MAX_LINE_WIDTH)



    #adjusting that only the surname of a player is presented.
    average_locs_and_count_df["playerName"] = average_locs_and_count_df["playerName"].apply(lambda x: str(x).split()[-1])
    #df_pass["pass_recipient_name"] = df_pass["pass_recipient_name"].apply(lambda x: str(x).split()[-1])



    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('#009991'))
    color = np.tile(color, (len(pass_between), 1))
    c_transparency = pass_between.pass_count / pass_between.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency



    pitch = pitch = Pitch(
        pitch_color= '#201D1D',
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    alpha = c_transparency
    norm = colors.Normalize(vmin=pass_between.total_xt.min(), vmax=pass_between.total_xt.max())

    # plot the arrows
    arrows = pitch.lines(pass_between.x,pass_between.y,pass_between.x_end,pass_between.y_end, linewidth=4,
                         color=cmap(norm(pass_between.total_xt.values)),
                         ax=ax, zorder=1, alpha=alpha)


    # Visualize the nodes using the new xt values
    nodes = pitch.scatter(average_locs_and_count_df.x, average_locs_and_count_df.y,
                          s = average_locs_and_count_df.marker_size, c = average_locs_and_count_df.xThreat_gen,
                          cmap = cmap, linewidth = 2.5, alpha = 1, zorder = 1, ax=ax)



    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0] - 0.5, pos_y[-1] + 0.5], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0] - 0.5, pos_x[-1] + 0.5], [y, y], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)

    for index, row in average_locs_and_count_df.iterrows():
        pitch.annotate(row.playerName, xy=(row.x, row.y), c='#132743', va='center', ha='center', size=6, ax=ax)


    ax.text(0.04, .04,
            'Node size = # of Passes' 
            '\n\nLine Transparency = Passes between', color='white', fontsize=8, ha='left', va='bottom',
            transform=ax.transAxes)


    return ax




def plot_player_hull_awayteam(player_df, ax, poly_edgecolor='#A0A0A0', poly_facecolor='#379A95', poly_alpha=0.3,
                              scatter_edgecolor='#0C0F0A', scatter_facecolor='#3c6e71', avg_marker_size=600,sd=int):


    pitch =  Pitch(pitch_type='opta')

    player_df['x'] = pitch.dim.right - player_df['x']
    player_df['y'] = pitch.dim.right - player_df['y']
    player_df['x_avg'] = pitch.dim.right - player_df['x_avg']
    player_df['y_avg'] = pitch.dim.right - player_df['y_avg']


    x_mean = player_df['x'].mean()
    y_mean = player_df['y'].mean()
    x_std = player_df['x'].std()
    y_std = player_df['y'].std()

    new_player_df = player_df[(np.abs(player_df['x'] - x_mean) < sd * x_std) & (np.abs(player_df['y'] - y_mean) < sd * y_std)]

    pitch = Pitch()
    # Draw convex hull polygon
    hull = pitch.convexhull(new_player_df.x, new_player_df.y)
    poly = pitch.polygon(hull, ax=ax, edgecolor=poly_edgecolor, facecolor=poly_facecolor, alpha=poly_alpha)

    # Draw scatter plot
    scatter = pitch.scatter(new_player_df.x, new_player_df.y, ax=ax, edgecolor=scatter_edgecolor, facecolor=scatter_facecolor,alpha=.30)

    # Draw average location marker
    pitch.scatter(new_player_df.x_avg, new_player_df.y_avg, ax=ax, edgecolor=scatter_edgecolor, facecolor=poly_facecolor,
                  s=avg_marker_size, marker='o', alpha=.20)

    # Add player initials as labels
    for i, row in new_player_df.iterrows():
        ax.text(row['x_avg'], row['y_avg'], row['initials'], fontsize=4, color='k', ha='center', va='center')

    return ax


def plot_team_passes_filter_away(ax,data,teamId,color,filter=None):
    df = data[data['is_open_play'] == True]
    df = data[data['teamId'] == teamId].copy()
    if filter:
        df = df.query(filter)


    pitch =  Pitch(pitch_type='opta')

    df['x'] = pitch.dim.right - df['x']
    df['y'] = pitch.dim.right - df['y']
    df['endX'] = pitch.dim.right - df['endX']
    df['endY'] = pitch.dim.right - df['endY']



    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    pitch.arrows(xstart=df.x, ystart=df.y, xend=df.endX, yend=df.endY, color=color, width=2, alpha=.8, ax=ax)

    return ax





def plot_clusters_event(ax, k, data, teamId, filter=None):
    df = data[data['teamId'] == teamId].copy()
    df = df[df['event_type'] == 'Pass']
    if filter:
        df = df.query(filter)

    columns_needed = ['x', 'y', 'endX', 'endY', 'pass_angle', 'distance', 'defensiveThird', 'midThird', 'finalThird']
    X = df.loc[:, columns_needed]
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    df['cluster'] = kmeans.predict(X)

    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)
    cluster_colors = ['#b18fcf', '#ef476f', '#26547c', '#e63946', '#ffb288', '#e5989b','#f4a261','#31572c','#fbf8cc']

    for cluster in range(k):
        cluster_df = df[df['cluster'] == cluster]
        color = cluster_colors[cluster % len(cluster_colors)]  # Ensure we cycle through colors

        # Calculate average pass for different thirds of the pitch within the cluster
        avg_pass_defensive = cluster_df[cluster_df['defensiveThird'] == 1][columns_needed].mean()
        avg_pass_middle = cluster_df[cluster_df['midThird'] == 1][columns_needed].mean()
        avg_pass_final = cluster_df[cluster_df['finalThird'] == 1][columns_needed].mean()

        # Plot individual passes within the cluster
        for _, row in cluster_df.iterrows():
            if row['defensiveThird'] == 1:
                avg_pass = avg_pass_defensive
            elif row['midThird'] == 1:
                avg_pass = avg_pass_middle
            else:
                avg_pass = avg_pass_final
            alpha = 0.08 if not row[columns_needed[:-3]].equals(avg_pass[columns_needed[:-3]]) else 0.4
            pitch.arrows(row['x'], row['y'], row['endX'], row['endY'], width=2.5, color=color, alpha=alpha,zorder = 10, ax=ax)

        # Plot the average passes for different thirds of the pitch within the cluster
        for avg_pass, third_color in zip([avg_pass_defensive, avg_pass_middle, avg_pass_final], ['#FF0000', '#00FF00', '#0000FF']):
            pitch.arrows(avg_pass['x'], avg_pass['y'], avg_pass['endX'], avg_pass['endY'], width=2.5, color=color, alpha=0.6,zorder = 11, ax=ax)

    return ax


def plot_event_clusters_away(ax, k, data, teamId, filter=None):
    df = data[data['teamId'] == teamId].copy()
    df = df[df['event_type'] == 'Pass']
    if filter:
        df = df.query(filter)

    pitch = Pitch(pitch_type='opta')

    df['x'] = pitch.dim.right - df['x']
    df['y'] = pitch.dim.right - df['y']
    df['endX'] = pitch.dim.right - df['endX']
    df['endY'] = pitch.dim.right - df['endY']

    columns_needed = ['x', 'y', 'endX', 'endY', 'pass_angle', 'distance', 'defensiveThird', 'midThird', 'finalThird']
    X = df.loc[:, columns_needed]
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    df['cluster'] = kmeans.predict(X)

    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)
    cluster_colors = ['#b18fcf', '#ef476f', '#26547c', '#e63946', '#ffb288', '#e5989b','#f4a261','#31572c','#fbf8cc']

    for cluster in range(k):
        cluster_df = df[df['cluster'] == cluster]
        color = cluster_colors[cluster % len(cluster_colors)]  # Ensure we cycle through colors

        # Calculate average pass for different thirds of the pitch within the cluster
        avg_pass_defensive = cluster_df[cluster_df['defensiveThird'] == 1][columns_needed].mean()
        avg_pass_middle = cluster_df[cluster_df['midThird'] == 1][columns_needed].mean()
        avg_pass_final = cluster_df[cluster_df['finalThird'] == 1][columns_needed].mean()

        # Plot individual passes within the cluster
        for _, row in cluster_df.iterrows():
            if row['defensiveThird'] == 1:
                avg_pass = avg_pass_defensive
            elif row['midThird'] == 1:
                avg_pass = avg_pass_middle
            else:
                avg_pass = avg_pass_final
            alpha = 0.08 if not row[columns_needed[:-3]].equals(avg_pass[columns_needed[:-3]]) else 0.4
            pitch.arrows(row['x'], row['y'], row['endX'], row['endY'], width=2.5, color=color, alpha=alpha,zorder = 10, ax=ax)

        # Plot the average passes for different thirds of the pitch within the cluster
        for avg_pass, third_color in zip([avg_pass_defensive, avg_pass_middle, avg_pass_final], ['#FF0000', '#00FF00', '#0000FF']):
            pitch.arrows(avg_pass['x'], avg_pass['y'], avg_pass['endX'], avg_pass['endY'], width=2.5, color=color, alpha=0.6,zorder = 11, ax=ax)

    return ax


def plot_team_passes_filter(ax,data,teamId,color,filter=None):
    df = data[data['is_open_play'] == True]
    df = data[data['teamId'] == teamId].copy()
    if filter:
        df = df.query(filter)

    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    pitch.arrows(xstart=df.x, ystart=df.y, xend=df.endX, yend=df.endY, color=color, width=2, alpha=.8, ax=ax)

    return ax


def plot_team_passes_filter_away(ax,data,teamId,color,filter=None):
    df = data[data['is_open_play'] == True]
    df = data[data['teamId'] == teamId].copy()
    if filter:
        df = df.query(filter)


    pitch =  Pitch(pitch_type='opta')

    df['x'] = pitch.dim.right - df['x']
    df['y'] = pitch.dim.right - df['y']
    df['endX'] = pitch.dim.right - df['endX']
    df['endY'] = pitch.dim.right - df['endY']



    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    pitch.arrows(xstart=df.x, ystart=df.y, xend=df.endX, yend=df.endY, color=color, width=2, alpha=.8, ax=ax)

    return ax


def plot_team_turnovers(ax, data, teamId, c):
    data = data[(data['teamId'] == teamId) & data['turnover']]
    df = data.copy()

    # df = df[df['is_pass_into_box']]
    pitch = VerticalPitch(
        pitch_color='#201D1D',
        half=False,
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=6,
        pad_right=8,
        pad_left=8,

        line_color='black')
    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y
    for x in pos_x[1:-1]:
        ax.plot([pos_y[0], pos_y[-1]], [x, x], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([y, y], [pos_x[0], pos_x[-1]], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)

    try:
        pitch.scatter(x=df.x, y=df.y, color=c, s=20, zorder=5, ax=ax, marker='o', alpha=0.8,
                      linewidths=1.6)
    except AttributeError as e:
        print(f"Error occurred while plotting turnovers: {e}")

    try:
        pitch.scatter(x=df[df['turnover_followed_by_shot']==True].x, y=df[df['turnover_followed_by_shot']==True].y, color='#f07167', s=20, zorder=6, ax=ax, marker='o', alpha=0.8,
                      linewidths=1.6)
    except AttributeError as e:
        print(f"Error occurred while plotting additional data: {e}")

    return ax


def plot_team_passes_filter(ax,data,teamId,color,filter=None):
    df = data[data['is_open_play'] == True]
    df = data[data['teamId'] == teamId].copy()
    if filter:
        df = df.query(filter)

    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    pitch.arrows(xstart=df.x, ystart=df.y, xend=df.endX, yend=df.endY, color=color, width=2, alpha=.8, ax=ax)

    return ax




#%%
def plot_pass_map_fulltime_subs_xT_away(ax, data,teamId:int, minute_start:int, minute_end:int, passes:int, touches:int,min_size:int,max_size:int,cmap_name ='coolwarm'):
    """Parameters:
    ax (Matplotlib Axes object): Axes object to plot the map on
    teamId (int): The Opta ID of the team to plot the map for
    data (dict): A dictionary containing Opta event data"""


    data = data.copy()


    data = data[data['teamId'] == teamId]
    player_pass_counts = data[data['isTouch'] == True]['playerName'].value_counts()
    qualified_players = player_pass_counts[player_pass_counts >= touches].index.tolist()
    data = data[data['playerName'].isin(qualified_players)]


    cmap = plt.get_cmap(cmap_name)


    def get_passes_df(events_dict):
        df = pd.DataFrame(events_dict)
        # create receiver column based on the next event
        # this will be correct only for successfull passes
        df["pass_recipient"] = df["playerName"].shift(-1)
        # filter only passes

        passes_ids = df.index[df['event_type'] == 'Pass']
        df_passes = df.loc[
            passes_ids, ["id", "minute", "x", "y", "endX", "endY", "teamId", "playerId", "playerName", "event_type",
                         "outcomeType", "pass_recipient",'isTouch','xThreat_gen']]

        return df_passes


    passes_df = get_passes_df(data)
    passes_df
    passes_df = passes_df[passes_df['outcomeType'] == 'Successful']

    minute_mask = (passes_df['minute'] >= minute_start) & (passes_df['minute'] <= minute_end)
    passes_df = passes_df.loc[minute_mask]


    average_locs_and_count_df = (passes_df.groupby('playerName')
                                 .agg({'x': ['mean'], 'y': ['mean', 'count']})
                                 .reset_index())

    average_locs_and_count_df.columns = ['playerName', 'x', 'y', 'count']


    substituted_players = data[data['event_type'].isin(['SubstitutionOn'])]['playerName'].unique()
    average_locs_and_count_df['is_substitute'] = average_locs_and_count_df['playerName'].isin(substituted_players)



    pass_between = passes_df.groupby(['playerName', 'pass_recipient']).agg(total_xt=('xThreat_gen', 'sum'),
                                                                           pass_count=('id', 'count')).reset_index()

    pass_between = pass_between[pass_between['pass_count'] >= passes]


    pass_between = pass_between.merge(average_locs_and_count_df[['playerName', 'x', 'y']], left_on='playerName', right_on='playerName')
    pass_between = pass_between.merge(average_locs_and_count_df[['playerName', 'x', 'y']], left_on='pass_recipient', right_on='playerName',
                                      suffixes=['', '_end'])

    player_xt_df = passes_df.groupby('playerName')['xThreat_gen'].sum().reset_index()

    average_locs_and_count_df = pd.merge(average_locs_and_count_df, player_xt_df, on='playerName')

    touches=data.groupby('playerName')['isTouch'].sum().reset_index()

    average_locs_and_count_df = pd.merge(average_locs_and_count_df, touches, on='playerName')

    MAX_MARKER_SIZE = max_size
    MIN_MARKER_SIZE = min_size

    min_count = average_locs_and_count_df['count'].min()
    max_count = average_locs_and_count_df['count'].max()

    average_locs_and_count_df['marker_size'] = ((average_locs_and_count_df['isTouch'] / max_count) * MAX_MARKER_SIZE
                                                + (1 - average_locs_and_count_df['isTouch'] / max_count) * MIN_MARKER_SIZE)


    MIN_LINE_WIDTH = 3.25
    MAX_LINE_WIDTH = 6.25

    pass_between['width'] = pass_between.pass_count / pass_between.pass_count.max() * MAX_LINE_WIDTH
    pass_between['width'] = pass_between['width'].clip(lower=MIN_LINE_WIDTH)


    pitch = Pitch(pitch_type='opta')


    pass_between['x'] = pitch.dim.right - pass_between['x']
    pass_between['y'] = pitch.dim.right - pass_between['y']
    pass_between['x_end'] = pitch.dim.right - pass_between['x_end']
    pass_between['y_end'] = pitch.dim.right - pass_between['y_end']
    average_locs_and_count_df['x'] = pitch.dim.right - average_locs_and_count_df['x']
    average_locs_and_count_df['y'] = pitch.dim.right - average_locs_and_count_df['y']


    average_locs_and_count_df['alpha'] = 0.4 + 0.6 * (average_locs_and_count_df['count'] / average_locs_and_count_df['count'].max())



    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('#009991'))
    color = np.tile(color, (len(pass_between), 1))
    c_transparency = pass_between.pass_count / pass_between.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency




    pitch = pitch = Pitch(
        pitch_color= '#201D1D',
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    alpha = c_transparency



    norm = colors.Normalize(vmin=pass_between.total_xt.min(), vmax=pass_between.total_xt.max())

    # plot the arrows
    arrows = pitch.lines(pass_between.x,pass_between.y,pass_between.x_end,pass_between.y_end, linewidth=pass_between.width,
                         color=cmap(norm(pass_between.total_xt.values)),
                         ax=ax, zorder=1, alpha=alpha,alpha_start=.01)



    average_locs_and_count_subs = average_locs_and_count_df[average_locs_and_count_df['is_substitute']==True]
    average_locs_and_count_starters = average_locs_and_count_df[average_locs_and_count_df['is_substitute']==False]


    # Visualize the nodes using the new xt values
    nodes = pitch.scatter(average_locs_and_count_starters.x, average_locs_and_count_starters.y,
                          c = average_locs_and_count_starters.xThreat_gen, s = average_locs_and_count_starters.marker_size,
                          cmap = cmap,  alpha=average_locs_and_count_starters.alpha,edgecolor='#353535', zorder = 1,
                          marker='o',
                          ax=ax)

    if not average_locs_and_count_subs.empty:
        nodes = pitch.scatter(average_locs_and_count_subs.x, average_locs_and_count_subs.y,
                              c = average_locs_and_count_subs.xThreat_gen, s = average_locs_and_count_subs.marker_size,
                              cmap = cmap,  alpha=average_locs_and_count_subs.alpha,edgecolor='#353535', zorder = 1,
                              marker= '8',
                              ax=ax)


    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0] - 0.5, pos_y[-1] + 0.5], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0] - 0.5, pos_x[-1] + 0.5], [y, y], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)

    # for index, row in average_locs_and_count_df.iterrows():
    #     pitch.annotate(row.playerName, xy=(row.x, row.y), c='#132743', va='center', ha='center', size=8, ax=ax)

    for index, row in average_locs_and_count_df.iterrows():
        name = row.playerName
        initials = ''.join([name[0] for name in name.split()])
        pitch.annotate(initials, xy=(row.x, row.y), c='k', va='center', ha='center', size=4, ax=ax)

    ax.text(0.04, 0.04, 'Node Transparency = # of Passes\n'
                        '\nLine Transparency = Passes between'
                        ,
            color='#edede9', fontsize=6, ha='left', va='bottom',
            transform=ax.transAxes)

    return ax


def plot_pass_map_fulltime_subs_xT(ax, data,teamId:int, minute_start:int, minute_end:int, passes:int, touches:int,min_size:int,max_size:int,cmap_name ='coolwarm'):
    """Parameters:
    ax (Matplotlib Axes object): Axes object to plot the map on
    teamId (int): The Opta ID of the team to plot the map for
    data (dict): A dictionary containing Opta event data"""

    data = data.copy()

    data =data[data['teamId'] == teamId]
    player_pass_counts = data[data['isTouch'] == True]['playerName'].value_counts()
    qualified_players = player_pass_counts[player_pass_counts >= touches].index.tolist()
    data = data[data['playerName'].isin(qualified_players)]


    cmap = plt.get_cmap(cmap_name)


    def get_passes_df(events_dict):
        df = pd.DataFrame(events_dict)
        # create receiver column based on the next event
        # this will be correct only for successfull passes
        df["pass_recipient"] = df["playerName"].shift(-1)
        # filter only passes

        passes_ids = df.index[df['event_type'] == 'Pass']
        df_passes = df.loc[
            passes_ids, ["id", "minute", "x", "y", "endX", "endY", "teamId", "playerId", "playerName", "event_type",
                         "outcomeType", "pass_recipient",'isTouch','xThreat_gen']]

        return df_passes


    passes_df = get_passes_df(data)
    passes_df
    passes_df = passes_df[passes_df['outcomeType'] == 'Successful']

    minute_mask = (passes_df['minute'] >= minute_start) & (passes_df['minute'] <= minute_end)
    passes_df = passes_df.loc[minute_mask]


    average_locs_and_count_df = (passes_df.groupby('playerName')
                                 .agg({'x': ['mean'], 'y': ['mean', 'count']})
                                 .reset_index())

    average_locs_and_count_df.columns = ['playerName', 'x', 'y', 'count']


    substituted_players = data[data['event_type'].isin(['SubstitutionOn'])]['playerName'].unique()
    average_locs_and_count_df['is_substitute'] = average_locs_and_count_df['playerName'].isin(substituted_players)



    pass_between = passes_df.groupby(['playerName', 'pass_recipient']).agg(total_xt=('xThreat_gen', 'sum'),
                                                                           pass_count=('id', 'count')).reset_index()

    pass_between = pass_between[pass_between['pass_count'] >= passes]


    pass_between = pass_between.merge(average_locs_and_count_df[['playerName', 'x', 'y']], left_on='playerName', right_on='playerName')
    pass_between = pass_between.merge(average_locs_and_count_df[['playerName', 'x', 'y']], left_on='pass_recipient', right_on='playerName',
                                      suffixes=['', '_end'])

    player_xt_df = passes_df.groupby('playerName')['xThreat_gen'].sum().reset_index()

    average_locs_and_count_df = pd.merge(average_locs_and_count_df, player_xt_df, on='playerName')

    touches=data.groupby('playerName')['isTouch'].sum().reset_index()

    average_locs_and_count_df = pd.merge(average_locs_and_count_df, touches, on='playerName')

    MAX_MARKER_SIZE = max_size
    MIN_MARKER_SIZE = min_size

    min_count = average_locs_and_count_df['count'].min()
    max_count = average_locs_and_count_df['count'].max()

    average_locs_and_count_df['marker_size'] = ((average_locs_and_count_df['isTouch'] / max_count) * MAX_MARKER_SIZE
                                                + (1 - average_locs_and_count_df['isTouch'] / max_count) * MIN_MARKER_SIZE)


    MIN_LINE_WIDTH = 3.25
    MAX_LINE_WIDTH = 8.25

    pass_between['width'] = pass_between.pass_count / pass_between.pass_count.max() * MAX_LINE_WIDTH
    pass_between['width'] = pass_between['width'].clip(lower=MIN_LINE_WIDTH)


    average_locs_and_count_df['alpha'] = 0.3 + 0.6 * (average_locs_and_count_df['count'] / average_locs_and_count_df['count'].max())



    MIN_TRANSPARENCY = 0.4
    color = np.array(to_rgba('#009991'))
    color = np.tile(color, (len(pass_between), 1))
    c_transparency = pass_between.pass_count / pass_between.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency




    pitch = pitch = Pitch(
        pitch_color= '#201D1D',
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    alpha = c_transparency



    norm = colors.Normalize(vmin=pass_between.total_xt.min(), vmax=pass_between.total_xt.max())

    # plot the arrows
    arrows = pitch.lines(pass_between.x,pass_between.y,pass_between.x_end,pass_between.y_end, linewidth=pass_between.width,
                         color=cmap(norm(pass_between.total_xt.values)),
                         ax=ax, zorder=1, alpha=alpha,alpha_start=.01)



    average_locs_and_count_subs = average_locs_and_count_df[average_locs_and_count_df['is_substitute']==True]
    average_locs_and_count_starters = average_locs_and_count_df[average_locs_and_count_df['is_substitute']==False]


    # Visualize the nodes using the new xt values
    nodes = pitch.scatter(average_locs_and_count_starters.x, average_locs_and_count_starters.y,
                          c = average_locs_and_count_starters.xThreat_gen, s = average_locs_and_count_starters.marker_size,
                          cmap = cmap,  alpha=average_locs_and_count_starters.alpha,edgecolor='#353535', zorder = 1,
                          marker='o',
                          ax=ax)

    if not average_locs_and_count_subs.empty:
        nodes = pitch.scatter(average_locs_and_count_subs.x, average_locs_and_count_subs.y,
                          c = average_locs_and_count_subs.xThreat_gen, s = average_locs_and_count_subs.marker_size,
                          cmap = cmap,  alpha=average_locs_and_count_subs.alpha,edgecolor='#353535', zorder = 1,
                          marker= '8',
                          ax=ax)


    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0] - 0.5, pos_y[-1] + 0.5], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0] - 0.5, pos_x[-1] + 0.5], [y, y], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)

    # for index, row in average_locs_and_count_df.iterrows():
    #     pitch.annotate(row.playerName, xy=(row.x, row.y), c='#132743', va='center', ha='center', size=8, ax=ax)

    for index, row in average_locs_and_count_df.iterrows():
        name = row.playerName
        initials = ''.join([name[0] for name in name.split()])
        pitch.annotate(initials, xy=(row.x, row.y), c='k', va='center', ha='center', size=4, ax=ax)

    ax.text(0.96, 0.06, 'Node Transparency = # of Passes'
                        '\nLine Transparency = Passes between'
                        ,
            color='#edede9', fontsize=6, ha='right', va='bottom',
            transform=ax.transAxes)

    return ax


def plot_pass_map_minute_xT_grid(ax, data,teamId:int, minute_start:int, minute_end:int, passes:int, touches:int,cmap_name='coolwarm'):
    """Parameters:
    ax (Matplotlib Axes object): Axes object to plot the map on
    teamId (int): The Opta ID of the team to plot the map for
    data (dict): A dictionary containing Opta event data"""

    data = data.copy()
    data =data[data['teamId'] == teamId]
    player_pass_counts = data[data['isTouch'] == True]['playerName'].value_counts()
    qualified_players = player_pass_counts[player_pass_counts >= touches].index.tolist()
    data = data[data['playerName'].isin(qualified_players)]


    cmap = plt.get_cmap(cmap_name)

    def get_passes_df(events_dict):
        df = pd.DataFrame(events_dict)
        # create receiver column based on the next event
        # this will be correct only for successfull passes
        df["pass_recipient"] = df["playerName"].shift(-1)
        # filter only passes

        passes_ids = df.index[df['event_type'] == 'Pass']
        df_passes = df.loc[
            passes_ids, ["id", "minute", "x", "y", "endX", "endY", "teamId", "playerId", "playerName", "event_type",
                         "outcomeType", "pass_recipient",'isTouch','xThreat_gen']]

        return df_passes


    passes_df = get_passes_df(data)
    passes_df
    passes_df = passes_df[passes_df['outcomeType'] == 'Successful']

    minute_mask = (passes_df['minute'] >= minute_start) & (passes_df['minute'] <= minute_end)
    # df_successful = df_successful.loc[(period_mask & minute_mask)]
    passes_df = passes_df.loc[minute_mask]

    average_locs_and_count_df = (passes_df.groupby('playerName')
                                 .agg({'x': ['mean'], 'y': ['mean', 'count']})
                                 .reset_index())

    average_locs_and_count_df.columns = ['playerName', 'x', 'y', 'count']

    substituted_players = data[data['event_type'].isin(['SubstitutionOn'])]['playerName'].unique()
    average_locs_and_count_df['is_substitute'] = average_locs_and_count_df['playerName'].isin(substituted_players)


    pass_between = passes_df.groupby(['playerName', 'pass_recipient']).agg(total_xt=('xThreat_gen', 'sum'),
                                                                           pass_count=('id', 'count')).reset_index()

    pass_between = pass_between[pass_between['pass_count'] >= passes]


    pass_between = pass_between.merge(average_locs_and_count_df[['playerName', 'x', 'y']], left_on='playerName', right_on='playerName')
    pass_between = pass_between.merge(average_locs_and_count_df[['playerName', 'x', 'y']], left_on='pass_recipient', right_on='playerName',
                                      suffixes=['', '_end'])

    player_xt_df = passes_df.groupby('playerName')['xThreat_gen'].sum().reset_index()

    average_locs_and_count_df = pd.merge(average_locs_and_count_df, player_xt_df, on='playerName')

    touches=data.groupby('playerName')['isTouch'].sum().reset_index()

    average_locs_and_count_df = pd.merge(average_locs_and_count_df, touches, on='playerName')

    # MAX_MARKER_SIZE = 80
    # MIN_MARKER_SIZE = 20
    #
    # min_count = average_locs_and_count_df['count'].min()
    # max_count = average_locs_and_count_df['count'].max()
    #
    # average_locs_and_count_df['marker_size'] = ((average_locs_and_count_df['isTouch'] / max_count) * MAX_MARKER_SIZE
    #                                             + (1 - average_locs_and_count_df['isTouch'] / max_count) * MIN_MARKER_SIZE)
    #

    MIN_LINE_WIDTH = 3.25
    MAX_LINE_WIDTH = 6.25

    pass_between['width'] = pass_between.pass_count / pass_between.pass_count.max() * MAX_LINE_WIDTH
    pass_between['width'] = pass_between['width'].clip(lower=MIN_LINE_WIDTH)

    average_locs_and_count_df['alpha'] = 0.3 + 0.7 * (average_locs_and_count_df['count'] / average_locs_and_count_df['count'].max())



    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('#009991'))
    color = np.tile(color, (len(pass_between), 1))
    c_transparency = pass_between.pass_count / pass_between.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency




    pitch = pitch = Pitch(
        pitch_color= '#201D1D',
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    alpha = c_transparency



    norm = colors.Normalize(vmin=pass_between.total_xt.min(), vmax=pass_between.total_xt.max())

    # plot the arrows
    arrows = pitch.lines(pass_between.x,pass_between.y,pass_between.x_end,pass_between.y_end, linewidth=pass_between.width,
                         color=cmap(norm(pass_between.total_xt.values)),
                         ax=ax, zorder=1, alpha=alpha,alpha_start=.01)


    average_locs_and_count_subs = average_locs_and_count_df[average_locs_and_count_df['is_substitute']==True]
    average_locs_and_count_starters = average_locs_and_count_df[average_locs_and_count_df['is_substitute']==False]


    # Visualize the nodes using the new xt values
    nodes = pitch.scatter(average_locs_and_count_starters.x, average_locs_and_count_starters.y,
                          c = average_locs_and_count_starters.xThreat_gen, s = 200,
                          cmap = cmap,  alpha=average_locs_and_count_starters.alpha,edgecolor='#353535', zorder = 1,
                          marker='o',
                          ax=ax)

    if not average_locs_and_count_subs.empty:
        nodes = pitch.scatter(average_locs_and_count_subs.x, average_locs_and_count_subs.y,
                              c = average_locs_and_count_subs.xThreat_gen, s = 200,
                              cmap = cmap,  alpha=average_locs_and_count_subs.alpha,edgecolor='#353535', zorder = 1,
                              marker= '8',
                              ax=ax)



    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0] - 0.5, pos_y[-1] + 0.5], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0] - 0.5, pos_x[-1] + 0.5], [y, y], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)

    for index, row in average_locs_and_count_df.iterrows():
        name = row.playerName
        initials = ''.join([name[0] for name in name.split()])
        pitch.annotate(initials, xy=(row.x, row.y), c='k', va='center', ha='center', size=4, ax=ax)



    return ax



def plot_pass_map_minute_xT_away_grid(ax, data,teamId:int, minute_start:int, minute_end:int, passes:int, touches:int,min_size,max_size,cmap_name='coolwarm'):
    """Parameters:
    ax (Matplotlib Axes object): Axes object to plot the map on
    teamId (int): The Opta ID of the team to plot the map for
    data (dict): A dictionary containing Opta event data"""

    data = data.copy()
    data =data[data['teamId'] == teamId]

    player_pass_counts = data[data['isTouch'] == True]['playerName'].value_counts()
    qualified_players = player_pass_counts[player_pass_counts >= touches].index.tolist()
    data = data[data['playerName'].isin(qualified_players)]

    cmap = plt.get_cmap(cmap_name)

    def get_passes_df(events_dict):
        df = pd.DataFrame(events_dict)
        # create receiver column based on the next event
        # this will be correct only for successfull passes
        df["pass_recipient"] = df["playerName"].shift(-1)
        # filter only passes

        passes_ids = df.index[df['event_type'] == 'Pass']
        df_passes = df.loc[
            passes_ids, ["id", "minute", "x", "y", "endX", "endY", "teamId", "playerId", "playerName", "event_type",
                         "outcomeType", "pass_recipient",'isTouch','xThreat_gen']]

        return df_passes


    passes_df = get_passes_df(data)
    pitch = Pitch(pitch_type='opta')


    passes_df = passes_df[passes_df['outcomeType'] == 'Successful']
    passes_df['x'] = pitch.dim.right - passes_df['x']
    passes_df['y'] = pitch.dim.right - passes_df['y']
    passes_df['endX'] = pitch.dim.right - passes_df['endX']
    passes_df['endY'] = pitch.dim.right - passes_df['endY']

    minute_mask = (passes_df['minute'] >= minute_start) & (passes_df['minute'] <= minute_end)
    # df_successful = df_successful.loc[(period_mask & minute_mask)]
    passes_df = passes_df.loc[minute_mask]


    # subs = passes_df[passes_df['e']=='']



    average_locs_and_count_df = (passes_df.groupby('playerName')
                                 .agg({'x': ['mean'], 'y': ['mean', 'count']})
                                 .reset_index())

    average_locs_and_count_df.columns = ['playerName', 'x', 'y', 'count']

    substituted_players = data[data['event_type'].isin(['SubstitutionOn'])]['playerName'].unique()
    average_locs_and_count_df['is_substitute'] = average_locs_and_count_df['playerName'].isin(substituted_players)



    pass_between = passes_df.groupby(['playerName', 'pass_recipient']).agg(total_xt=('xThreat_gen', 'sum'),
                                                                           pass_count=('id', 'count')).reset_index()

    pass_between = pass_between[pass_between['pass_count'] >= passes]






    pass_between = pass_between.merge(average_locs_and_count_df[['playerName', 'x', 'y']], left_on='playerName', right_on='playerName')
    pass_between = pass_between.merge(average_locs_and_count_df[['playerName', 'x', 'y']], left_on='pass_recipient', right_on='playerName',
                                      suffixes=['', '_end'])

    player_xt_df = passes_df.groupby('playerName')['xThreat_gen'].sum().reset_index()

    average_locs_and_count_df = pd.merge(average_locs_and_count_df, player_xt_df, on='playerName')


    touches=data.groupby('playerName')['isTouch'].sum().reset_index()

    average_locs_and_count_df = pd.merge(average_locs_and_count_df, touches, on='playerName')

    # MAX_MARKER_SIZE = max_size
    # MIN_MARKER_SIZE = min_size
    #
    # min_count = average_locs_and_count_df['count'].min()
    # max_count = average_locs_and_count_df['count'].max()
    #
    # average_locs_and_count_df['marker_size'] = ((average_locs_and_count_df['isTouch'] / max_count) * MAX_MARKER_SIZE
    #                                             + (1 - average_locs_and_count_df['isTouch'] / max_count) * MIN_MARKER_SIZE)


    MIN_LINE_WIDTH = 3.25
    MAX_LINE_WIDTH = 8.25

    pass_between['width'] = pass_between.pass_count / pass_between.pass_count.max() * MAX_LINE_WIDTH
    pass_between['width'] = pass_between['width'].clip(lower=MIN_LINE_WIDTH)


    average_locs_and_count_df['alpha'] = 0.3 + 0.7 * (average_locs_and_count_df['count'] / average_locs_and_count_df['count'].max())




    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('#009991'))
    color = np.tile(color, (len(pass_between), 1))
    c_transparency = pass_between.pass_count / pass_between.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency




    pitch = pitch = Pitch(
        pitch_color= '#201D1D',
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    alpha = c_transparency



    norm = colors.Normalize(vmin=pass_between.total_xt.min(), vmax=pass_between.total_xt.max())

    # plot the arrows
    arrows = pitch.lines(pass_between.x,pass_between.y,pass_between.x_end,pass_between.y_end, linewidth=pass_between.width,
                         color=cmap(norm(pass_between.total_xt.values)),
                         ax=ax, zorder=1, alpha=alpha,alpha_start=.01)


    average_locs_and_count_subs = average_locs_and_count_df[average_locs_and_count_df['is_substitute']==True]
    average_locs_and_count_starters = average_locs_and_count_df[average_locs_and_count_df['is_substitute']==False]


    # Visualize the nodes using the new xt values
    nodes = pitch.scatter(average_locs_and_count_starters.x, average_locs_and_count_starters.y,
                          c = average_locs_and_count_starters.xThreat_gen, s = 200,
                          cmap = cmap,  alpha=average_locs_and_count_starters.alpha,edgecolor='#353535', zorder = 1,
                          marker='o',
                          ax=ax)

    if not average_locs_and_count_subs.empty:
        nodes = pitch.scatter(average_locs_and_count_subs.x, average_locs_and_count_subs.y,
                              c = average_locs_and_count_subs.xThreat_gen, s = 200,
                              cmap = cmap,  alpha=average_locs_and_count_subs.alpha,edgecolor='#353535', zorder = 1,
                              marker= '8',
                              ax=ax)




    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0] - 0.5, pos_y[-1] + 0.5], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0] - 0.5, pos_x[-1] + 0.5], [y, y], color='black', ls='dashed', zorder=0, lw=0.8, alpha=0.85)

    for index, row in average_locs_and_count_df.iterrows():
        name = row.playerName
        initials = ''.join([name[0] for name in name.split()])
        pitch.annotate(initials, xy=(row.x, row.y), c='k', va='center', ha='center', size=4, ax=ax)

    return ax




def plot_xT_players_barplot(ax,data, teamid: int, event_type: str, color: str):
    subset = data[(data['teamId'] == teamid) & (data['event_type'] == event_type)]
    subset = subset.copy()
    subset.loc[:, 'last_name'] = subset['playerName'].str.split().str[-1]

    # Group by last_name and compute the sum of xThreat_gen for each group
    grouped = subset.groupby(['last_name'])['xThreat_gen'].sum().reset_index()
    grouped = grouped.sort_values(by='xThreat_gen', ascending=True)

    # Create a horizontal bar plot of the xThreat_gen values, with player last names as the y-axis labels
    x = grouped['last_name']
    y = grouped['xThreat_gen']
    ax.barh(x, y, color=color, edgecolor='#151313',alpha=.8 ,hatch='\\\\',zorder=3,linewidth=2)


    # Customize the plot
    ax.set_xlabel('xThreat', color='#E1D3D3',size=6)
    ax.set_ylabel('Player', color='#E1D3D3',size=6)
    # ax.set_title(f'xT via {event_type}', color='#E1D3D3',size=12)
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax.yaxis.grid(True, linestyle='--', which='both', color='grey', alpha=.25)

    # Set the tick colors
    ax.tick_params(axis='x', colors='#E1D3D3')
    ax.tick_params(axis='y', colors='#E1D3D3')
    ax.set_ylabel('')


    return ax


def get_passes_df(events_dict):
    df = pd.DataFrame(events_dict)
    # create receiver column based on the next event
    # this will be correct only for successfull passes
    df["pass_recipient"] = df["playerName"].shift(-1)
    # filter only passes

    passes_ids = df.index[(df['event_type'] == 'Pass') | (df['event_type'] == 'Carry')]
    df_passes = df.loc[
        passes_ids, ["id","minute", "x", "y", "endX", "endY", "teamId", "playerId","playerName", "event_type", "outcomeType","pass_recipient",'xThreat_gen','xThreat']]

    return df_passes

def plot_xT_rec_barplot_teams(ax,data, event_type: str, colors: list):

    data =data.copy()

    passes = get_passes_df(data)

    subset = passes[passes['event_type']==event_type]

    # Compute the xThreat for each recipient
    xThreat_received = subset.groupby(['pass_recipient','teamId'])['xThreat_gen'].sum().reset_index()

    # Extract the last names of the recipients
    xThreat_received['last_name'] = xThreat_received['pass_recipient'].str.split().str[-1]

    # Filter out recipients with xThreat less than or equal to 0.1
    xThreat_received = xThreat_received[xThreat_received['xThreat_gen'] > 0.1]

    # Sort by xThreat and create a horizontal bar plot
    xThreat_received = xThreat_received.sort_values(by='xThreat_gen', ascending=True)
    # fig, ax = plt.subplots(figsize=(8, 6), dpi=700)

    # Define the color for each team
    colors_dict = dict(zip(xThreat_received['teamId'].unique(), colors))
    colors_list = [colors_dict[team_id] for team_id in xThreat_received['teamId']]

    ax.barh(xThreat_received['last_name'], xThreat_received['xThreat_gen'], color=colors_list, edgecolor='#201D1D', hatch='\\\\',zorder=3)

    # Customize the plot
    ax.set_xlabel('xThreat Receive', color='#E1D3D3',size=6)
    ax.set_ylabel('Player', color='#E1D3D3',size=6)
    # ax.set_title(f'xT via {event_type}', color='#E1D3D3')
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('')
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax.yaxis.grid(True, linestyle='--', which='both', color='grey', alpha=.25)

    # Set the tick colors
    ax.tick_params(axis='x', colors='#E1D3D3')
    ax.tick_params(axis='y', colors='#E1D3D3',size=6)
    return ax





def plot_xT_heatmap(ax,data,teamId,color):

    data_xT = data.copy()
    data_xT = data_xT[data_xT['teamId']==teamId]
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax = ax)
    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y


    data_xT = data_xT.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_xT = data_xT.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))


    data_grouped = data_xT.groupby(['bins_x', 'bins_y'], as_index=False)['xThreat_gen'].sum()
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['xThreat_gen'].min() / data_grouped['xThreat_gen'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1
    return ax



def plot_xT_heatmap_away(ax,data,teamId,color):

    data_xT = data.copy()
    data_xT = data_xT[data_xT['teamId']==teamId]
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax = ax)
    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    data_xT['x'] = pitch.dim.right - data_xT['x']
    data_xT['y'] = pitch.dim.right - data_xT['y']
    data_xT['endX'] = pitch.dim.right - data_xT['endX']
    data_xT['endY'] = pitch.dim.right - data_xT['endY']

    data_xT = data_xT.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_xT = data_xT.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))



    data_grouped = data_xT.groupby(['bins_x', 'bins_y'], as_index=False)['xThreat_gen'].sum()
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['xThreat_gen'].min() / data_grouped['xThreat_gen'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * .6,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.35,
                    ec='None'
                )
            counter += 1
    return ax




def plot_keypasses_team_opp_half(ax,data,teamId):

    data = data[data['is_open_play'] == True]
    data = data[data['teamId'] == teamId]
    data = data[data['event_type'] == 'Pass']
    # Draw pitch
    pitch = VerticalPitch(
        pitch_color='#201D1D',
        half=True,
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    # Draw penalty box arc

    # Draw dashed lines
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y
    for x in pos_x[1:-1]:
        ax.plot([pos_y[0], pos_y[-1]], [x, x], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([y, y], [pos_x[0], pos_x[-1]], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)

    # pitch.arrows(data.x, data.y, data.endX, data.endY, width=2,
    #              headwidth=4.5, headlength=4.5, color='#395253', alpha=.2, label='Pass', zorder=1, ax=ax)

    pitch.arrows(data[data['key_pass'] == True]
                 .x, data[data['key_pass'] == True]
                 .y, data[data['key_pass'] == True]
                 .endX, data[data['key_pass'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#3c6e71', label='Key Pass', alpha=.8, zorder=2, ax=ax)

    pitch.arrows(data[data['assist'] == True]
                 .x, data[data['assist'] == True]
                 .y, data[data['assist'] == True]
                 .endX, data[data['assist'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#db5461', alpha=.8, zorder=3, label='Assist', ax=ax)

    pitch.arrows(data[data['pre_assist'] == True]
                 .x, data[data['pre_assist'] == True]
                 .y, data[data['pre_assist'] == True]
                 .endX, data[data['pre_assist'] == True]
                 .endY, width=2,
                 headwidth=4.5, headlength=4.5, color='#f4a261', label='Pre Assist', alpha=.8, zorder=3, ax=ax)

    # Add legend
    legend = ax.legend(ncol=4, loc='upper center', fontsize=6, markerscale=.0002, labelcolor='w', framealpha=.0,
                       bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    return ax


def plot_player_heatmap(ax, data,playerName, color,sd):
    data = data.copy()
    # data = data[data['is_open_play'] & data['isTouch']]
    # data=

    data= data[data['is_open_play']==True]

    # data = data[(data['is_open_play'] == True) & (data['isTouch'] == True)]

    data = data[(data['playerName'] == playerName) & (data['outcomeType'] == 'Successful')]

    # data_touch = data[data['isTouch']==True].copy()

    data_touch = data.copy()
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    sigma = sd # set the standard deviation of the Gaussian kernel

    data_touch['isTouch_gaussian_filter'] = ndi.gaussian_filter(data_touch['isTouch'], sigma)

    data_touch = data_touch.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_touch = data_touch.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_touch.groupby(['bins_x', 'bins_y'], as_index=False)['isTouch_gaussian_filter'].sum()

    # data_grouped = data_touch.groupby(['bins_x', 'bins_y']).size().reset_index(name='isTouch')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['isTouch_gaussian_filter'].min() / data_grouped['isTouch_gaussian_filter'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * 1,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.2,
                    ec='None'
                )
            counter += 1


    #ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax




def plot_deep_comp_team(ax, data, teamId, passColor, carryColor, radius):
    data = data[(data['teamId'] == teamId) & (data['outcomeType'] == 'Successful') & (data['is_open_play'] == True)]

    data = data[(data['event_type'] == 'Pass') | (data['event_type'] == 'Carry')]

    # Filter data by event type
    passes = data[data['event_type'] == 'Pass']
    carries = data[data['event_type'] == 'Carry']

    # Calculate distances
    data['initialDistancefromgoal'] = np.sqrt(((100 - data['x']) ** 2) + ((50 - data['y']) ** 2))
    passes['finalDistancefromgoal'] = np.sqrt(((100 - passes['endX']) ** 2) + ((50 - passes['endY']) ** 2))
    carries['finalDistancefromgoalcarry'] = np.sqrt(((100 - carries['endX']) ** 2) + ((50 - carries['endY']) ** 2))

    # Calculate if a pass or carry is a deep completion
    data['deepCompletion'] = np.where(
        ((passes['finalDistancefromgoal'] <= radius) & (data['initialDistancefromgoal'] >= radius)) | (
                (carries['finalDistancefromgoalcarry'] <= radius) & (data['initialDistancefromgoal'] >= radius)),
        'True', 'False')

    pitch = VerticalPitch(
        pitch_color='#201D1D',
        half=True,
        linewidth=2.5,
        corner_arcs=True,
        pitch_type='opta',
        goal_type='box',
        pad_top=10,
        line_color='black')
    pitch.draw(ax=ax)

    # function for semicircle
    def semicircle(r, h, k):
        x0 = h - r  # determine x start
        x1 = h + r  # determine x finish
        x = np.linspace(x0, x1, 10000)  # many points to solve for y

        # use numpy for array solving of the semicircle equation
        y = k - np.sqrt(r ** 2 - (x - h) ** 2)
        return x, y

    x_circle, y_circle = semicircle(radius, 50, 100)
    ax.plot(x_circle, y_circle, ls='--', color='#e5dddc', lw=.75)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y
    for x in pos_x[1:-1]:
        ax.plot([pos_y[0], pos_y[-1]], [x, x], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)
    for y in pos_y[1:-1]:
        ax.plot([y, y], [pos_x[0], pos_x[-1]], color='black', ls='dashed', zorder=0, lw=0.3, alpha=0.85)

    pitch.lines(data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Pass')]['x'],
                data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Pass')]['y'],
                data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Pass')]['endX'],
                data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Pass')]['endY'],
                comet=True, color=passColor, ax=ax, transparent=True, lw=2, alpha=0.6, label='Pass')

    pitch.scatter(data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Pass')]['endX'],
                  data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Pass')]['endY'],
                  c=passColor, s=40, zorder=4, ax=ax, marker='o', alpha=0.8, linewidths=2)

    pitch.lines(data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Carry')]['x'],
                data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Carry')]['y'],
                data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Carry')]['endX'],
                data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Carry')]['endY'],
                comet=True, color=carryColor, ax=ax, transparent=True, lw=2, alpha=0.6, label='Carry')

    pitch.scatter(data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Carry')]['endX'],
                  data[(data['deepCompletion'] == 'True') & (data['event_type'] == 'Carry')]['endY'],
                  c=carryColor, s=40, zorder=4, ax=ax, marker='o', alpha=0.8, linewidths=2)

    legend = ax.legend(ncol=4, loc='upper center', fontsize=6, markerscale=.0002, labelcolor='w', framealpha=.0,
                       bbox_to_anchor=(.5, 0.03))
    legend.get_frame().set_facecolor('#D1D1D1')

    return ax


def plot_team_heatmap_away(ax, data, teamid, color, sd):
    data = data.copy()
    # data = data[data['is_open_play'] & data['isTouch']]
    # data=

    data = data[data['is_open_play'] == True]



    data = data[(data['teamId'] == teamid) & (data['outcomeType'] == 'Successful')]

    # data_touch = data[data['isTouch']==True].copy()

    data_touch = data.copy()
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)


    data_touch['x'] = pitch.dim.right - data_touch['x']
    data_touch['y'] = pitch.dim.right - data_touch['y']
    data_touch['endX'] = pitch.dim.right - data_touch['endX']
    data_touch['endY'] = pitch.dim.right - data_touch['endY']


    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    sigma = sd  # set the standard deviation of the Gaussian kernel

    data_touch['isTouch_gaussian_filter'] = ndi.gaussian_filter(data_touch['isTouch'], sigma)

    data_touch = data_touch.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_touch = data_touch.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_touch.groupby(['bins_x', 'bins_y'], as_index=False)['isTouch_gaussian_filter'].sum()

    # data_grouped = data_touch.groupby(['bins_x', 'bins_y']).size().reset_index(name='isTouch')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['isTouch_gaussian_filter'].min() / data_grouped[
                'isTouch_gaussian_filter'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * 1,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.2,
                    ec='None'
                )
            counter += 1

    #ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax

def plot_team_heatmap(ax, data,teamid, color,sd):
    data = data.copy()
    # data = data[data['is_open_play'] & data['isTouch']]
    # data=

    data= data[data['is_open_play']==True]

    # data = data[(data['is_open_play'] == True) & (data['isTouch'] == True)]

    data = data[(data['teamId'] == teamid) & (data['outcomeType'] == 'Successful')]

    # data_touch = data[data['isTouch']==True].copy()

    data_touch = data.copy()
    pitch = Pitch(
        pitch_type='opta',
        goal_type='box',
        linewidth=2.5,
        line_color='black',
        half=False
    )

    pitch.draw(ax=ax)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    sigma = sd # set the standard deviation of the Gaussian kernel

    data_touch['isTouch_gaussian_filter'] = ndi.gaussian_filter(data_touch['isTouch'], sigma)

    data_touch = data_touch.assign(bins_x=lambda x: pd.cut(x.x, bins=pos_x))
    data_touch = data_touch.assign(bins_y=lambda x: pd.cut(x.y, bins=pos_y))

    data_grouped = data_touch.groupby(['bins_x', 'bins_y'], as_index=False)['isTouch_gaussian_filter'].sum()

    # data_grouped = data_touch.groupby(['bins_x', 'bins_y']).size().reset_index(name='isTouch')
    data_grouped['left_x'] = data_grouped['bins_x'].apply(lambda x: x.left)
    data_grouped['right_x'] = data_grouped['bins_x'].apply(lambda x: x.right)
    data_grouped['left_y'] = data_grouped['bins_y'].apply(lambda x: x.left)
    data_grouped['right_y'] = data_grouped['bins_y'].apply(lambda x: x.right)

    # Here we can get the positional dimensions
    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y

    for x in pos_x[1:-1]:
        ax.plot([x, x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y, y], color='#000000', ls='dashed', zorder=0, lw=0.3)

    counter = 1
    for index_y, y in enumerate(pos_y):
        for index_x, x in enumerate(pos_x):
            try:
                lower_y = pos_y[index_y]
                lower_x = pos_x[index_x]
                upper_y = pos_y[index_y + 1]
                upper_x = pos_x[index_x + 1]
            except:
                continue
            condition_bounds = (data_grouped['left_x'] >= lower_x) & (data_grouped['right_x'] <= upper_x) & (
                    data_grouped['left_y'] >= lower_y) & (data_grouped['right_y'] <= upper_y)
            alpha = data_grouped[condition_bounds]['isTouch_gaussian_filter'].min() / data_grouped['isTouch_gaussian_filter'].max()
            if alpha > 0:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color=color,
                    zorder=0,
                    alpha=alpha * 1,
                    ec='None'
                )
            else:
                ax.fill_between(
                    x=[lower_x, upper_x],
                    y1=lower_y,
                    y2=upper_y,
                    color='grey',
                    zorder=0,
                    alpha=.2,
                    ec='None'
                )
            counter += 1


    #ax.legend(ncol=5, loc='lower center', fontsize=5.5, bbox_to_anchor=[.48, -.1])
    ax.set_facecolor("#201D1D")
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return ax




#%%

#%%
