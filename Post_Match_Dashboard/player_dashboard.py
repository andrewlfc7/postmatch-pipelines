import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import mplsoccer to demo creating a pitch on gridsearch
from mplsoccer import Pitch
from mplsoccer import VerticalPitch
import sqlite3
from highlight_text import fig_text, ax_text
from ast import literal_eval
from unidecode import unidecode
from google.cloud import storage
from io import BytesIO
import yaml

import requests
import bs4
import json
from PIL import Image
import urllib
from Football_Analysis_Tools import fotmob_visuals as fotmobvis
from Football_Analysis_Tools import  whoscored_visuals as whovis
from Football_Analysis_Tools import whoscored_data_engineering as who_eng
import os
from sqlalchemy import create_engine
import datetime
import pytz

user = os.environ['PGUSER']
passwd = os.environ['PGPASSWORD']
host = os.environ['PGHOST']
port = os.environ['PGPORT']
db = os.environ['PGDATABASE']
db_url = f'postgresql://{user}:{passwd}@{host}:{port}/{db}'

engine = create_engine(db_url)

conn = engine.connect()



eastern = pytz.timezone('US/Eastern')
today = datetime.datetime.now(eastern).date()
today = today.strftime('%Y-%m-%d')



shots_query =f"""
SELECT * FROM fotmob_shots_data WHERE match_date = '{today}' AND ("teamId" = 8650 OR "match_id" IN (SELECT "match_id" FROM fotmob_shots_data WHERE "teamId" = 8650))

"""

query =f"""
SELECT * FROM opta_event_data WHERE match_date = '{today}' AND ("teamId" = 26 OR "match_id" IN (SELECT "match_id" FROM opta_event_data WHERE "teamId" = 26))
"""



key_file_path = '/Post_Match_Dashboard/postmatch-key'

# Check if the file exists
if os.path.exists(key_file_path):
    # Read the contents of the file
    with open(key_file_path, 'r') as key_file:
        key_data = key_file.read()

    # Assuming the key data is in JSON format
    key_json = json.loads(key_data)

    # Use key_json as needed in your code
    # For example, you can access individual keys like key_json['key_name']

    # Assuming GOOGLE_APPLICATION_CREDENTIALS is expected to contain the service account key path
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path
else:
    print("Error: Scraper key file not found at", key_file_path)


shots_data = pd.read_sql(shots_query, conn)

data = pd.read_sql(query, conn)

data['playerName'] = data['playerName'].replace('Unknown', pd.NA)
data = data.dropna(subset=['playerId', 'playerName'])

Fotmob_matchID = shots_data['match_id'].iloc[0]



# boolean columns
bool_cols = ['isTouch',
             'is_open_play',
             'is_progressive',
             'is_pass_into_box',
             'won_possession',
             'key_pass',
             'assist',
             'FinalThirdPasses',
             'pre_assist',
             'switch']

# convert boolean columns to boolean values
for col in bool_cols:
    data[col] = data[col].astype(bool)



# def get_opp_name(match_id):
#     response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
#     data = json.loads(response.content)
#     general = data['general']
#     Hteam = general['homeTeam']
#     Ateam = general['awayTeam']
#     Hteam = Hteam['name']
#     Ateam = Ateam['name']
#     return Hteam + " " + "vs" + " " + Ateam
#
#
# match_name = get_match_name(Fotmob_matchID)
#


data['qualifiers'] = [literal_eval(x) for x in data['qualifiers']]
data['satisfiedEventsTypes'] = [literal_eval(x) for x in data['satisfiedEventsTypes']]
CrossPasses_set = set(['passCrossAccurate', 'passCrossInaccurate'])
data = data.copy()
data['is_cross'] = False
for index, row in enumerate(data['satisfiedEventsTypes']):
    set_element = set(row)
    if len(CrossPasses_set.intersection(set_element)) > 0:
        data.at[index, 'is_cross'] = True


#%%
# data[(~data['is_open_play']) & (data['assist']) & (data['key_pass'])]

#%%
# filter for rows where openplay is True
openplay_data = data.loc[data['is_open_play'] == True]

# filter for rows where assist and keypass are True but openplay is False
assist_keypass_data = data.loc[(data['assist'] == True) & (data['key_pass'] == True) & (data['is_open_play'] == False)]

# combine the two filtered dataframes
combined_data = pd.concat([openplay_data, assist_keypass_data])

# sort the combined dataframe by index
combined_data = combined_data.sort_index()

#%%
data=combined_data
#%%



def get_passes_df(events_dict):
    df = pd.DataFrame(events_dict)
    # create receiver column based on the next event
    # this will be correct only for successfull passes
    df["pass_recipient"] = df["playerName"].shift(-1)
    # filter only passes

    passes_ids = df.index[df['event_type'] == 'Pass']
    df_passes = df.loc[
        passes_ids,  ["id","minute", "x", "y", "endX", "endY", "teamId", "playerId","playerName", "event_type", "outcomeType","pass_recipient",'switch','pre_assist','assist','FinalThirdPasses','key_pass','is_open_play','is_progressive','is_cross']]

    return df_passes



passes_df = data[data['teamId']==26]

#%%
passes_df = get_passes_df(passes_df)
#%%
def find_offensive_actions(events_df):
    """ Return dataframe of in-play offensive actions from event data.
    Function to find all in-play offensive actions within a whoscored-style events dataframe (single or multiple
    matches), and return as a new dataframe.
    Args:
        events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
    Returns:
        pandas.DataFrame: whoscored-style dataframe of offensive actions.
    """

    # Define and filter offensive events
    offensive_actions = ['TakeOn', 'MissedShots', 'SavedShot', 'Goal', 'Carry','ShotOnPost']
    offensive_action_df = events_df[events_df['event_type'].isin(offensive_actions)].reset_index(drop=True)

    # Filter for passes with assists or pre-assists
    pass_df = events_df[(events_df['event_type'] == 'Pass') & ((events_df['assist'] == True) | (events_df['pre_assist'] == True))]

    # Concatenate offensive actions and passes with assists or pre-assists
    offensive_actions_df = pd.concat([offensive_action_df, pass_df]).reset_index(drop=True)

    return offensive_actions_df

offensive_actions = find_offensive_actions(data)
defensive_actions = who_eng.find_defensive_actions(data)

from unidecode import unidecode



centerbacks = [
    "Ibrahima Konaté",
    "Virgil van Dijk",
    "Joël Matip",
    "Jarell Quansah"
]


fullbacks = [
    "Andy Robertson",
    "Trent Alexander-Arnold",
    "Joe Gomez",
    "Conor Bradley",
    "Kostas Tsimikas"
]

midfielders = [
    "Alexis Mac Allister",
    "Dominik Szoboszlai",
    "Curtis Jones",
    "Harvey Elliott",
    "Wataru Endo",
    "Ryan Gravenberch",
    "Stefan Bajcetic",
    "Bobby Clark"
]


forwards = [
    "Luis Díaz",
    "Mohamed Salah",
    "Cody Gakpo",
    "Diogo Jota",
    "Darwin Núñez"


]

# Check if players played in the match
def check_players_in_match(players, data):
    players_in_match = []
    for player in players:
        if player in data['playerName'].values:
            players_in_match.append(player)
    return players_in_match


centerbacks_in_match = check_players_in_match(centerbacks, data)
fullbacks_in_match = check_players_in_match(fullbacks, data)
midfielders_in_match = check_players_in_match(midfielders, data)
forwards_in_match = check_players_in_match(forwards, data)




# Fetch player data from the API
response = requests.get('https://www.fotmob.com/api/teams?id=8650&ccode3=USA_MA')
player_id_data = json.loads(response.content)
player_data = player_id_data['squad']

# Initialize an empty list to store all players
all_players = []

# Titles of interest
titles_of_interest = ['keepers', 'defenders', 'midfielders', 'attackers']

# Iterate over player_data which is a list
for players_info in player_data:
    title = players_info.get('title', '')
    if title in titles_of_interest:
        players = players_info.get('members', [])
        all_players.extend(players)

# Create a DataFrame from the combined player data
df_players = pd.DataFrame(all_players)[['id', 'name']]


player_id_dict = dict(zip(df_players['name'], df_players['id']))



def get_player_id(name):
    # First, try to find player ID with accents
    player_id = player_id_dict.get(name)

    if player_id is None:
        # If not found, split the name into first and last names
        names = name.split()
        if len(names) == 2:
            first_name, last_name = names

            # Try searching with both first and last names
            matching_first_names = df_players[df_players['name'].str.contains(first_name, case=False, na=False)]
            matching_last_names = df_players[df_players['name'].str.contains(last_name, case=False, na=False)]

            if len(matching_first_names) == 1:
                player_id = matching_first_names.iloc[0]['id']
            elif len(matching_last_names) == 1:
                player_id = matching_last_names.iloc[0]['id']

        # If still not found, remove accents and try again
        if player_id is None:
            normalized_name = unidecode(name)
            player_id = player_id_dict.get(normalized_name)

            if player_id is None:
                # If still not found, try to find shorter names without accents
                for player_name in player_id_dict.keys():
                    if (len(player_name) >= len(name) and
                            unidecode(player_name).startswith(unidecode(name))):
                        player_id = player_id_dict.get(player_name)
                        break

    return player_id



# for player_name in centerbacks:
#     player_id = get_player_id(player_name)
#     try:
#         player_logo_path = f'Post_Match_Dashboard/Data/player_image/{player_id}.png'
#         club_icon = Image.open(player_logo_path).convert('RGBA')
#     except FileNotFoundError as e:
#         print(f"Could not find image for player: {player_name}")
#         continue
#     fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
#     gs = fig.add_gridspec(ncols=3, nrows=3)  # change nrows from 2 to 3
#     fig.set_facecolor("#201D1D")
#
#     # create subplots using gridspec
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])  # add new subplot
#     ax4 = fig.add_subplot(gs[1, 0])
#     ax5 = fig.add_subplot(gs[1, 1])  # add new subplot
#     ax6 = fig.add_subplot(gs[1, 2])  # add new subplot
#
#     axes = [ax1, ax2, ax3, ax4, ax5, ax6]
#
#     # apply modifications to all subplots
#     for ax in axes:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xlabel('')
#         ax.set_ylabel('')
#         ax.grid(False)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         ax.set_facecolor("#201D1D")
#         ax.axis('off')
#
#     ax1.set_title('Offensive Actions', color='#e1dbd6', fontsize=16, pad=10)
#     ax4.set_title('Ball Carries', color='#e1dbd6', fontsize=16, pad=10)
#     ax3.set_title('Defensive Actions', color='#e1dbd6', fontsize=16, pad=10)
#     ax2.set_title('Territory map', color='#e1dbd6', fontsize=16, pad=10)
#     ax5.set_title('Pass Map', color='#e1dbd6', fontsize=16, pad=10)
#     ax6.set_title('Passes Receive ', color='#e1dbd6', fontsize=16, pad=10)
#
#
#     whovis.plot_players_offensive_actions_opta(ax1, player_name, offensive_actions, color='#1f8e98')
#     whovis.plot_carry_player_opta(ax4, player_name, data)
#
#     whovis.plot_players_defensive_actions_opta(ax3, player_name, defensive_actions, color='#1f8e98')
#
#     whovis.plot_player_heatmap(ax2, data, player_name, color='#1f8e98', sd=0)
#
#     whovis.plot_player_passmap_opta(ax5, player_name, data)
#
#     whovis.plot_player_passes_rec_opta(ax6, player_name, passes_df)
#
#
#     player_logo_path = f'Post_Match_Dashboard/Data/player_image/{player_id}.png'
#     club_icon = Image.open(player_logo_path).convert('RGBA')
#
#     logo_ax = fig.add_axes([0, .94, 0.12, 0.12], frameon=False)
#     logo_ax.imshow(club_icon)
#     logo_ax.set_xticks([])
#     logo_ax.set_yticks([])
#
#     fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')
#
#     # plt.savefig(
#     #     f"Post_Match_Dashboard/figures/playerdashboard{player_name}.png",
#     #     dpi=600,
#     #     bbox_inches="tight",
#     #     edgecolor="none",
#     #     transparent=False
#     # )
#     # Create a BytesIO object to store the figure
#     figure_buffer = BytesIO()
#
#     # Save the figure to the BytesIO object
#     plt.savefig(
#         figure_buffer,
#         format="png",  # Use the appropriate format for your figure
#         dpi=600,
#         bbox_inches="tight",
#         edgecolor="none",
#         transparent=False
#     )
#
#     # Reset the buffer position to the beginning
#     figure_buffer.seek(0)
#
#     # Initialize Google Cloud Storage client and get the bucket
#     storage_client = storage.Client()
#     bucket_name = "postmatch-dashboards"
#     bucket = storage_client.get_bucket(bucket_name)
#
#     # Specify the blob path within the bucket
#     blob_path = f"figures/{today}/players/playerdashboard{player_name}.png"
#
#     # Create a new Blob and upload the figure
#     blob = bucket.blob(blob_path)
#     blob.upload_from_file(figure_buffer, content_type="image/png")
#
#     # Close the BytesIO buffer
#     figure_buffer.close()
#
#



import matplotlib.pyplot as plt
from io import BytesIO
from google.cloud import storage

# Iterate over centerbacks
for player_name in centerbacks_in_match:
    # Create figure and subplots
    fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
    gs = fig.add_gridspec(ncols=3, nrows=3)
    fig.set_facecolor("#201D1D")

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # Modify subplots
    for ax in axes:
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
        ax.axis('off')

    # Set titles for subplots
    ax1.set_title('Offensive Actions', color='#e1dbd6', fontsize=16, pad=10)
    ax4.set_title('Ball Carries', color='#e1dbd6', fontsize=16, pad=10)
    ax3.set_title('Defensive Actions', color='#e1dbd6', fontsize=16, pad=10)
    ax2.set_title('Territory map', color='#e1dbd6', fontsize=16, pad=10)
    ax5.set_title('Pass Map', color='#e1dbd6', fontsize=16, pad=10)
    ax6.set_title('Passes Receive', color='#e1dbd6', fontsize=16, pad=10)

    # Plot data for each subplot
    whovis.plot_players_offensive_actions_opta(ax1, player_name, offensive_actions, color='#1f8e98')
    whovis.plot_carry_player_opta(ax4, player_name, data)
    whovis.plot_players_defensive_actions_opta(ax3, player_name, defensive_actions, color='#1f8e98')
    whovis.plot_player_heatmap(ax2, data, player_name, color='#1f8e98', sd=0)
    whovis.plot_player_passmap_opta(ax5, player_name, data)
    whovis.plot_player_passes_rec_opta(ax6, player_name, passes_df)

    # Set figure title
    fig.suptitle(f'{player_name} Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')

    # Save figure to BytesIO object
    figure_buffer = BytesIO()
    plt.savefig(
        figure_buffer,
        format="png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer.seek(0)

    # Upload figure to Google Cloud Storage
    storage_client = storage.Client()
    bucket_name = "postmatch-dashboards"
    bucket = storage_client.get_bucket(bucket_name)
    blob_path = f"figures/{today}/players/playerdashboard{player_name}.png"
    blob = bucket.blob(blob_path)
    blob.upload_from_file(figure_buffer, content_type="image/png")
    figure_buffer.close()



# for player_name in fullbacks:
#     player_id = get_player_id(player_name)
#     try:
#         player_logo_path = f'Post_Match_Dashboard/Data/player_image/{player_id}.png'
#         club_icon = Image.open(player_logo_path).convert('RGBA')
#     except FileNotFoundError as e:
#         print(f"Could not find image for player: {player_name}")
#         continue
#     fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
#     gs = fig.add_gridspec(ncols=3, nrows=3)  # change nrows from 2 to 3
#     fig.set_facecolor("#201D1D")
#
#     # create subplots using gridspec
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])  # add new subplot
#     ax4 = fig.add_subplot(gs[1, 0])
#     ax5 = fig.add_subplot(gs[1, 1])  # add new subplot
#     ax6 = fig.add_subplot(gs[1, 2])  # add new subplot
#
#     axes = [ax1, ax2, ax3, ax4, ax5, ax6]
#
#     # apply modifications to all subplots
#     for ax in axes:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xlabel('')
#         ax.set_ylabel('')
#         ax.grid(False)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         ax.set_facecolor("#201D1D")
#         ax.axis('off')
#
#     ax1.set_title('Offensive Actions', color='#e1dbd6', fontsize=16, pad=10)
#     ax4.set_title('Ball Carries', color='#e1dbd6', fontsize=16, pad=10)
#     ax3.set_title('Defensive Actions', color='#e1dbd6', fontsize=16, pad=10)
#     ax2.set_title('Territory map', color='#e1dbd6', fontsize=16, pad=10)
#     ax5.set_title('Pass Map', color='#e1dbd6', fontsize=16, pad=10)
#     ax6.set_title('Passes Receive ', color='#e1dbd6', fontsize=16, pad=10)
#
#
#     whovis.plot_players_offensive_actions_opta(ax1, player_name, offensive_actions, color='#1f8e98')
#     whovis.plot_carry_player_opta(ax4, player_name, data)
#
#     whovis.plot_players_defensive_actions_opta(ax3, player_name, defensive_actions, color='#1f8e98')
#
#     whovis.plot_player_heatmap(ax2, data, player_name, color='#1f8e98', sd=0)
#
#     whovis.plot_player_passmap_opta(ax5, player_name, data)
#
#     whovis.plot_player_passes_rec_opta(ax6, player_name, passes_df)
#
#
#     player_logo_path = f'Post_Match_Dashboard/Data/player_image/{player_id}.png'
#     club_icon = Image.open(player_logo_path).convert('RGBA')
#
#     logo_ax = fig.add_axes([0, .94, 0.12, 0.12], frameon=False)
#     logo_ax.imshow(club_icon)
#     logo_ax.set_xticks([])
#     logo_ax.set_yticks([])
#
#     fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')
#
#     # plt.savefig(
#     #     f"Post_Match_Dashboard/figures/playerdashboard{player_name}.png",
#     #     dpi=600,
#     #     bbox_inches="tight",
#     #     edgecolor="none",
#     #     transparent=False
#     # )
#
#     # Create a BytesIO object to store the figure
#     figure_buffer = BytesIO()
#
#     # Save the figure to the BytesIO object
#     plt.savefig(
#         figure_buffer,
#         format="png",  # Use the appropriate format for your figure
#         dpi=600,
#         bbox_inches="tight",
#         edgecolor="none",
#         transparent=False
#     )
#
#     # Reset the buffer position to the beginning
#     figure_buffer.seek(0)
#
#     # Initialize Google Cloud Storage client and get the bucket
#     storage_client = storage.Client()
#     bucket_name = "postmatch-dashboards"
#     bucket = storage_client.get_bucket(bucket_name)
#
#     # Specify the blob path within the bucket
#     blob_path = f"figures/{today}/players/playerdashboard{player_name}.png"
#
#     # Create a new Blob and upload the figure
#     blob = bucket.blob(blob_path)
#     blob.upload_from_file(figure_buffer, content_type="image/png")
#
#     # Close the BytesIO buffer
#     figure_buffer.close()


# Iterate over fullbacks
for player_name in fullbacks_in_match:
    fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
    gs = fig.add_gridspec(ncols=3, nrows=3)
    fig.set_facecolor("#201D1D")

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    for ax in axes:
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
        ax.axis('off')

    ax1.set_title('Offensive Actions', color='#e1dbd6', fontsize=16, pad=10)
    ax4.set_title('Ball Carries', color='#e1dbd6', fontsize=16, pad=10)
    ax3.set_title('Defensive Actions', color='#e1dbd6', fontsize=16, pad=10)
    ax2.set_title('Territory map', color='#e1dbd6', fontsize=16, pad=10)
    ax5.set_title('Pass Map', color='#e1dbd6', fontsize=16, pad=10)
    ax6.set_title('Passes Receive ', color='#e1dbd6', fontsize=16, pad=10)

    whovis.plot_players_offensive_actions_opta(ax1, player_name, offensive_actions, color='#1f8e98')
    whovis.plot_carry_player_opta(ax4, player_name, data)
    whovis.plot_players_defensive_actions_opta(ax3, player_name, defensive_actions, color='#1f8e98')
    whovis.plot_player_heatmap(ax2, data, player_name, color='#1f8e98', sd=0)
    whovis.plot_player_passmap_opta(ax5, player_name, data)
    whovis.plot_player_passes_rec_opta(ax6, player_name, passes_df)

    fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')

    figure_buffer = BytesIO()
    plt.savefig(
        figure_buffer,
        format="png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer.seek(0)

    storage_client = storage.Client()
    bucket_name = "postmatch-dashboards"
    bucket = storage_client.get_bucket(bucket_name)

    blob_path = f"figures/{today}/players/playerdashboard{player_name}.png"

    blob = bucket.blob(blob_path)
    blob.upload_from_file(figure_buffer, content_type="image/png")

    figure_buffer.close()



# for player_name in midfielders:
#     player_id = get_player_id(player_name)
#     try:
#         player_logo_path = f'Post_Match_Dashboard/Data/player_image/{player_id}.png'
#         club_icon = Image.open(player_logo_path).convert('RGBA')
#     except FileNotFoundError as e:
#         print(f"Could not find image for player: {player_name}")
#         continue
#     fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
#     gs = fig.add_gridspec(ncols=3, nrows=3)  # change nrows from 2 to 3
#     fig.set_facecolor("#201D1D")
#
#     # create subplots using gridspec
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])  # add new subplot
#     ax4 = fig.add_subplot(gs[1, 0])
#     ax5 = fig.add_subplot(gs[1, 1])  # add new subplot
#     ax6 = fig.add_subplot(gs[1, 2])  # add new subplot
#
#     axes = [ax1, ax2, ax3, ax4, ax5, ax6]
#
#     # apply modifications to all subplots
#     for ax in axes:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xlabel('')
#         ax.set_ylabel('')
#         ax.grid(False)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         ax.set_facecolor("#201D1D")
#         ax.axis('off')
#
#
#     ax1.set_title('Offensive Actions', color ='#e1dbd6'  , fontsize=16, pad=10)
#     ax4.set_title('Ball Carries',color ='#e1dbd6'  , fontsize=16, pad=10)
#     ax3.set_title('Defensive Actions',color ='#e1dbd6'  , fontsize=16, pad=10)
#     ax2.set_title('Territory map', color ='#e1dbd6'  ,fontsize=16, pad=10)
#     ax5.set_title('Pass Map', color ='#e1dbd6'  ,fontsize=16, pad=10)
#     ax6.set_title('Passes Receive ', color ='#e1dbd6'  ,fontsize=16, pad=10)
#
#
#
#
#     whovis.plot_players_offensive_actions_opta(ax1,player_name,offensive_actions,color='#1f8e98')
#     whovis.plot_carry_player_opta(ax4,player_name,data)
#     whovis.plot_players_defensive_actions_opta(ax3,player_name,defensive_actions,color='#1f8e98')
#     whovis.plot_player_heatmap(ax2, data,player_name,color='#1f8e98',sd=0)
#     whovis.plot_player_passmap_opta(ax5,player_name,data)
#     whovis.plot_player_passes_rec_opta(ax6,player_name,passes_df)
#
#
#
#     player_logo_path = f'Post_Match_Dashboard/Data/player_image/{player_id}.png'
#     club_icon = Image.open(player_logo_path).convert('RGBA')
#
#     logo_ax = fig.add_axes([0, .94, 0.12, 0.12], frameon=False)
#     logo_ax.imshow(club_icon)
#     logo_ax.set_xticks([])
#     logo_ax.set_yticks([])
#
#     fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')
#
#     # plt.savefig(
#     #     f"Post_Match_Dashboard/figures/playerdashboard{player_name}.png",
#     #     dpi=600,
#     #     bbox_inches="tight",
#     #     edgecolor="none",
#     #     transparent=False
#     # )
#
#     # Create a BytesIO object to store the figure
#     figure_buffer = BytesIO()
#
#     # Save the figure to the BytesIO object
#     plt.savefig(
#         figure_buffer,
#         format="png",  # Use the appropriate format for your figure
#         dpi=600,
#         bbox_inches="tight",
#         edgecolor="none",
#         transparent=False
#     )
#
#     # Reset the buffer position to the beginning
#     figure_buffer.seek(0)
#
#     # Initialize Google Cloud Storage client and get the bucket
#     storage_client = storage.Client()
#     bucket_name = "postmatch-dashboards"
#     bucket = storage_client.get_bucket(bucket_name)
#
#     # Specify the blob path within the bucket
#     blob_path = f"figures/{today}/players/playerdashboard{player_name}.png"
#
#     # Create a new Blob and upload the figure
#     blob = bucket.blob(blob_path)
#     blob.upload_from_file(figure_buffer, content_type="image/png")
#
#     # Close the BytesIO buffer
#     figure_buffer.close()
#


# Iterate over midfielders
for player_name in midfielders_in_match:
    fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
    gs = fig.add_gridspec(ncols=3, nrows=3)
    fig.set_facecolor("#201D1D")

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    for ax in axes:
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
        ax.axis('off')

    ax1.set_title('Offensive Actions', color ='#e1dbd6'  , fontsize=16, pad=10)
    ax4.set_title('Ball Carries',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax3.set_title('Defensive Actions',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax2.set_title('Territory map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax5.set_title('Pass Map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax6.set_title('Passes Receive ', color ='#e1dbd6'  ,fontsize=16, pad=10)

    whovis.plot_players_offensive_actions_opta(ax1,player_name,offensive_actions,color='#1f8e98')
    whovis.plot_carry_player_opta(ax4,player_name,data)
    whovis.plot_players_defensive_actions_opta(ax3,player_name,defensive_actions,color='#1f8e98')
    whovis.plot_player_heatmap(ax2, data,player_name,color='#1f8e98',sd=0)
    whovis.plot_player_passmap_opta(ax5,player_name,data)
    whovis.plot_player_passes_rec_opta(ax6,player_name,passes_df)

    fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')

    figure_buffer = BytesIO()
    plt.savefig(
        figure_buffer,
        format="png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer.seek(0)

    storage_client = storage.Client()
    bucket_name = "postmatch-dashboards"
    bucket = storage_client.get_bucket(bucket_name)

    blob_path = f"figures/{today}/players/playerdashboard{player_name}.png"

    blob = bucket.blob(blob_path)
    blob.upload_from_file(figure_buffer, content_type="image/png")

    figure_buffer.close()


#
# for player_name in forwards:
#     player_id = get_player_id(player_name)
#     fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
#     gs = fig.add_gridspec(ncols=3, nrows=3)  # change nrows from 2 to 3
#     fig.set_facecolor("#201D1D")
#
#     # create subplots using gridspec
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax3 = fig.add_subplot(gs[0, 2])  # add new subplot
#     ax4 = fig.add_subplot(gs[1, 0])
#     ax5 = fig.add_subplot(gs[1, 1])  # add new subplot
#     ax6 = fig.add_subplot(gs[1, 2])  # add new subplot
#
#     axes = [ax1, ax2, ax3, ax4, ax5, ax6]
#
#     # apply modifications to all subplots
#     for ax in axes:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_xlabel('')
#         ax.set_ylabel('')
#         ax.grid(False)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         ax.set_facecolor("#201D1D")
#         ax.axis('off')
#
#
#     ax1.set_title('Shot Map', color ='#e1dbd6'  , fontsize=16, pad=10)
#     ax4.set_title('Ball Carries',color ='#e1dbd6'  , fontsize=16, pad=10)
#     ax3.set_title('Offensive Actions',color ='#e1dbd6'  , fontsize=16, pad=10)
#     ax2.set_title('Territory map', color ='#e1dbd6'  ,fontsize=16, pad=10)
#     ax5.set_title('Pass Map', color ='#e1dbd6'  ,fontsize=16, pad=10)
#     ax6.set_title('Passes Receive ', color ='#e1dbd6'  ,fontsize=16, pad=10)
#
#
#
#     fotmobvis.plot_player_shotmap(ax1,Fotmob_matchID,player_name)
#
#     whovis.plot_carry_player_opta(ax4,player_name,data)
#
#     whovis.plot_players_defensive_actions_opta(ax3,player_name,offensive_actions,color='#1f8e98')
#
#     whovis.plot_player_passmap_opta(ax5,player_name,data)
#
#     whovis.plot_player_passes_rec_opta(ax6,player_name,passes_df)
#     whovis.plot_player_heatmap(ax2, data,player_name,color='#1f8e98',sd=0)
#
#
#     player_logo_path = f'Post_Match_Dashboard/Data/player_image/{player_id}.png'
#     club_icon = Image.open(player_logo_path).convert('RGBA')
#
#     logo_ax = fig.add_axes([0, .94, 0.12, 0.12], frameon=False)
#     logo_ax.imshow(club_icon)
#     logo_ax.set_xticks([])
#     logo_ax.set_yticks([])
#
#     fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')
#
#     # plt.savefig(
#     #     f"Post_Match_Dashboard/figures/playerdashboard{player_name}.png",
#     #     dpi=600,
#     #     bbox_inches="tight",
#     #     edgecolor="none",
#     #     transparent=False
#     # )
#
#     # Create a BytesIO object to store the figure
#     figure_buffer = BytesIO()
#
#     # Save the figure to the BytesIO object
#     plt.savefig(
#         figure_buffer,
#         format="png",  # Use the appropriate format for your figure
#         dpi=600,
#         bbox_inches="tight",
#         edgecolor="none",
#         transparent=False
#     )
#
#     # Reset the buffer position to the beginning
#     figure_buffer.seek(0)
#
#     # Initialize Google Cloud Storage client and get the bucket
#     storage_client = storage.Client()
#     bucket_name = "postmatch-dashboards"
#     bucket = storage_client.get_bucket(bucket_name)
#
#     # Specify the blob path within the bucket
#     blob_path = f"figures/{today}/players/playerdashboard{player_name}.png"
#
#     # Create a new Blob and upload the figure
#     blob = bucket.blob(blob_path)
#     blob.upload_from_file(figure_buffer, content_type="image/png")
#
#     # Close the BytesIO buffer
#     figure_buffer.close()




# Iterate over forwards
for player_name in forwards_in_match:
    fig = plt.figure(figsize=(15, 13), constrained_layout=True, dpi=600)
    gs = fig.add_gridspec(ncols=3, nrows=3)
    fig.set_facecolor("#201D1D")

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    for ax in axes:
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
        ax.axis('off')

    ax1.set_title('Shot Map', color ='#e1dbd6'  , fontsize=16, pad=10)
    ax4.set_title('Ball Carries',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax3.set_title('Offensive Actions',color ='#e1dbd6'  , fontsize=16, pad=10)
    ax2.set_title('Territory map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax5.set_title('Pass Map', color ='#e1dbd6'  ,fontsize=16, pad=10)
    ax6.set_title('Passes Receive ', color ='#e1dbd6'  ,fontsize=16, pad=10)

    fotmobvis.plot_player_shotmap(ax1,Fotmob_matchID,player_name)
    whovis.plot_carry_player_opta(ax4,player_name,data)
    whovis.plot_players_defensive_actions_opta(ax3,player_name,offensive_actions,color='#1f8e98')
    whovis.plot_player_passmap_opta(ax5,player_name,data)
    whovis.plot_player_passes_rec_opta(ax6,player_name,passes_df)
    whovis.plot_player_heatmap(ax2, data,player_name,color='#1f8e98',sd=0)

    fig.suptitle(f'{player_name}  Post Match Dashboard', fontsize=24, color='#e1dbd6', ha='center')

    figure_buffer = BytesIO()
    plt.savefig(
        figure_buffer,
        format="png",
        dpi=600,
        bbox_inches="tight",
        edgecolor="none",
        transparent=False
    )
    figure_buffer.seek(0)

    storage_client = storage.Client()
    bucket_name = "postmatch-dashboards"
    bucket = storage_client.get_bucket(bucket_name)

    blob_path = f"figures/{today}/players/playerdashboard{player_name}.png"

    blob = bucket.blob(blob_path)
    blob.upload_from_file(figure_buffer, content_type="image/png")

    figure_buffer.close()



