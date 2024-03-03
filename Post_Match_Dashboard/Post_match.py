import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch
from mplsoccer import VerticalPitch
from highlight_text import fig_text, ax_text
import matplotlib

from matplotlib.colors import LinearSegmentedColormap

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from PIL import Image
import yaml
from Football_Analysis_Tools import whoscored_visuals as whovis

from Football_Analysis_Tools import odds
from Football_Analysis_Tools import fotmob_visuals as fmvis
import os
from google.cloud import storage
from io import BytesIO

import datetime

from sqlalchemy import create_engine

import datetime
import pytz

from utils import check_logo_existence, ax_logo,get_and_save_logo,mark_turnover_followed_by_shot,pass_angle
from ast import literal_eval


user = os.environ['PGUSER']
passwd = os.environ['PGPASSWORD']
host = os.environ['PGHOST']
port = os.environ['PGPORT']
db = os.environ['PGDATABASE']
db_url = f'postgresql://{user}:{passwd}@{host}:{port}/{db}'

engine = create_engine(db_url)




# # Set the time zone to Eastern Time
eastern = pytz.timezone('US/Eastern')
#
# # Get the current date in Eastern Time
today = datetime.datetime.now(eastern).date()
today = today.strftime('%Y-%m-%d')




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



conn = engine.connect()

shots_query =f"""
SELECT * FROM fotmob_shots_data WHERE match_date = '{today}' AND ("teamId" = 8650 OR "match_id" IN (SELECT "match_id" FROM fotmob_shots_data WHERE "teamId" = 8650))

"""


query =f"""
SELECT * FROM opta_event_data WHERE match_date = '{today}' AND ("teamId" = 26 OR "match_id" IN (SELECT "match_id" FROM opta_event_data WHERE "teamId" = 26))
"""


shots_data = pd.read_sql(shots_query, conn)

data = pd.read_sql(query, conn)



os.environ['GOOGLE_APPLICATION_CREDENTIALS']='Post_Match_Dashboard/careful-aleph-398521-f12755bcaea3.json'




data = data[data['is_open_play']==True]
data = data[data['outcomeType']=='Successful']

event_home_df = data[data['Venue']=='Home']
event_away_df = data[data['Venue']=='Away']

opta_matchID = data['match_id'].iloc[0]

opta_home_teamID = event_home_df['teamId'].iloc[0]
opta_away_teamID = event_away_df['teamId'].iloc[0]


Fotmob_matchID = shots_data['match_id'].iloc[0]

Fotmob_homeID = shots_data[shots_data['Venue']=='Home']
Fotmob_awayID = shots_data[shots_data['Venue']=='Away']
Fotmob_homeID = Fotmob_homeID['teamId'].iloc[0]
Fotmob_awayID = Fotmob_awayID['teamId'].iloc[0]



import requests
import json

def get_match_label(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
    data = json.loads(response.content)
    general = data['general']
    leagueName = general['leagueName']
    leagueRoundName = general['leagueRoundName']
    return leagueName + " " + leagueRoundName





def get_match_season(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
    data = json.loads(response.content)
    general = data['general']
    parentLeagueSeason=general['parentLeagueSeason']
    return parentLeagueSeason

match_season = get_match_season(Fotmob_matchID)

def get_match_date(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
    data = json.loads(response.content)
    general = data['general']
    matchDate = general['matchTimeUTCDate']
    matchDate = matchDate.split('T')[0]

    return matchDate

matchDate = get_match_date(Fotmob_matchID)

def get_match_name(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
    data = json.loads(response.content)
    general = data['general']
    Hteam = general['homeTeam']
    Ateam = general['awayTeam']
    Hteam = Hteam['name']
    Ateam = Ateam['name']
    return Hteam + " " + "vs" +" " + Ateam

match_name = get_match_name(Fotmob_matchID)

def get_match_score(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
    data = json.loads(response.content)
    match_score = data['header']['status']['scoreStr']
    return match_score


match_score = get_match_score(Fotmob_matchID)
match_label = get_match_label(Fotmob_matchID)



home = data[data['teamId']==opta_home_teamID]
away = data[data['teamId']==opta_away_teamID]


# colors = ["#29222c","#32283b","#3b2d4b","#44335a","#4c3869","#553e78","#5e4388","#674997","#704ea6"]

colors = ["#264653","#2a9d8f","#e9c46a","#f4a261","#e76f51"]

passnetwork_cmap = LinearSegmentedColormap.from_list('passnetwork', colors, N=250)

plt.register_cmap(cmap=passnetwork_cmap)


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



def are_colors_similar_shade(color1, color2, threshold=40):
    # Compare colors based on their RGB values
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:], 16)

    # Calculate color difference using Euclidean distance for grayscale
    gray1 = 0.2989 * r1 + 0.5870 * g1 + 0.1140 * b1
    gray2 = 0.2989 * r2 + 0.5870 * g2 + 0.1140 * b2

    return abs(gray1 - gray2) < threshold

# Get the unique team IDs
unique_team_ids = shots_data['teamId'].unique()

# Define the target color
target_color = '#1A6158'

# Iterate through unique team IDs
for team_id in unique_team_ids:
    team_color = shots_data[shots_data['teamId'] == team_id]['teamColor'].iloc[0]

    # Check if colors are similar in shade
    if are_colors_similar_shade(team_color, target_color):
        # Change the color for the team with teamid 8650
        if team_id == 8650:
            shots_data.loc[shots_data['teamId'] == team_id, 'teamColor'] = target_color



homecolor = shots_data[shots_data['Venue'] == 'Home']['teamColor'].iloc[0]
awaycolor = shots_data[shots_data['Venue'] == 'Away']['teamColor'].iloc[0]


data['qualifiers'] = [literal_eval(x) for x in data['qualifiers']]
data['satisfiedEventsTypes'] = [literal_eval(x) for x in data['satisfiedEventsTypes']]


pass_angles = pass_angle(data["x"], data["y"], data["endX"], data["endY"]).tolist()

data["pass_angle"] = pass_angles



check_logo_existence(away_id=Fotmob_awayID,home_id=Fotmob_homeID)

if check_logo_existence(away_id=Fotmob_awayID, home_id=Fotmob_homeID):
    print(f'Logos for teams {Fotmob_awayID} and {Fotmob_homeID} exist.')
else:
    print(f'One or both logos are missing. Fetching and saving...')
    if not os.path.exists(f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'):
        get_and_save_logo(Fotmob_awayID)
    if not os.path.exists(f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'):
        get_and_save_logo(Fotmob_homeID)




data = mark_turnover_followed_by_shot(data)


def get_pass_stats(data):
    data.loc[:, "playerName"] = data["playerName"].apply(lambda x: x.split()[-1])

    def get_passes_df(events_dict):
        df = pd.DataFrame(events_dict)

        # create receiver column based on the next event
        # this will be correct only for successful passes
        df["pass_recipient"] = df["playerName"].shift(-1)
        # filter only passes
        passes_ids = df.index[df['event_type'] == 'Pass']
        df_passes = df.loc[passes_ids, ["id", "minute", "x", "y", "endX", "endY", "teamId", "playerId",
                                        "playerName", "event_type", "outcomeType", "pass_recipient", 'isTouch',
                                        'xThreat_gen', 'is_progressive']]
        return df_passes

    passes_df = get_passes_df(data)
    passes_df = passes_df[passes_df['outcomeType'] == 'Successful']
    # passes_df = passes_df['playerName'].apply(lambda x: x.split()[-1])
    # passes_df = passes_df['pass_recipient'].apply(lambda x: x.split()[-1])
    pass_between = passes_df.groupby(['playerName', 'pass_recipient', 'is_progressive']).agg(
        total_xt=('xThreat_gen', 'sum'),
        pass_count=('id', 'count')).reset_index()

    progressive_passes = pd.DataFrame(data[data['is_progressive'] == True]['playerName'].value_counts())
    progressive_passes.columns = ['progressive_passes']

    most_passes = pd.DataFrame(data[data['event_type'] == 'Pass']['playerName'].value_counts())
    most_passes.columns = ['most_passes']

    highest_xt = pd.DataFrame(data.groupby('playerName')['xThreat_gen'].sum())
    highest_xt.columns = ['highest_xt']

    result = pass_between.join(most_passes, on='playerName')
    result = result.join(highest_xt, on='playerName')
    result = result.join(progressive_passes, on='playerName')

    return result

home_pass_stats = get_pass_stats(home)
away_pass_stats= get_pass_stats(away)
prog_rec_away=away_pass_stats.groupby(['pass_recipient'])['is_progressive'].sum().sort_values(ascending=False).reset_index().loc[0, 'pass_recipient']
xT_rec_away = away_pass_stats.groupby(['pass_recipient'])['total_xt'].sum().sort_values(ascending=False).reset_index().loc[0, 'pass_recipient']
prog_passes_away = away_pass_stats.groupby(['playerName'])['is_progressive'].sum().sort_values(ascending=False).reset_index().loc[0, 'playerName']
passes_away = away_pass_stats.groupby(['playerName'])['pass_count'].sum().sort_values(ascending=False).reset_index().loc[0, 'playerName']
xT_pass_away=away_pass_stats.groupby(['playerName'])['total_xt'].sum().sort_values(ascending=False).sort_values(ascending=False).reset_index().loc[0, 'playerName']
pass_rec_away = away_pass_stats['pass_recipient'].value_counts().sort_values(ascending=False).reset_index().iloc[0, 0]



most_pass_row_away = away_pass_stats.groupby(['playerName', 'pass_recipient'])['pass_count'].sum().sort_values(ascending=False).reset_index().iloc[0]
most_pass_player_away = most_pass_row_away['playerName']
most_pass_recipient_away = most_pass_row_away['pass_recipient']

most_pass_combo_away = f"{most_pass_player_away} -> {most_pass_recipient_away}"


highest_xt_row_away = away_pass_stats.groupby(['playerName', 'pass_recipient'])['total_xt'].sum().sort_values(ascending=False).reset_index().iloc[0]
highest_xt_player_away = highest_xt_row_away['playerName']
highest_xt_recipient_away = highest_xt_row_away['pass_recipient']


highest_xt_combo_away = f"{highest_xt_player_away} -> {highest_xt_recipient_away}"

prog_rec_home=home_pass_stats.groupby(['pass_recipient'])['is_progressive'].sum().sort_values(ascending=False).reset_index().loc[0, 'pass_recipient']
xT_rec_home = home_pass_stats.groupby(['pass_recipient'])['total_xt'].sum().sort_values(ascending=False).reset_index().loc[0, 'pass_recipient']
prog_passes_home = home_pass_stats.groupby(['playerName'])['is_progressive'].sum().sort_values(ascending=False).reset_index().loc[0, 'playerName']
passes_home = home_pass_stats.groupby(['playerName'])['pass_count'].sum().sort_values(ascending=False).reset_index().loc[0, 'playerName']
xT_pass_home=home_pass_stats.groupby(['playerName'])['total_xt'].sum().sort_values(ascending=False).sort_values(ascending=False).reset_index().loc[0, 'playerName']
pass_rec_home = home_pass_stats['pass_recipient'].value_counts().sort_values(ascending=False).reset_index().iloc[0, 0]



most_pass_row_home = home_pass_stats.groupby(['playerName', 'pass_recipient'])['pass_count'].sum().sort_values(ascending=False).reset_index().iloc[0]
most_pass_player_home = most_pass_row_home['playerName']
most_pass_recipient_home = most_pass_row_home['pass_recipient']

most_pass_combo_home = f"{most_pass_player_home} -> {most_pass_recipient_home}"


highest_xt_row_home = home_pass_stats.groupby(['playerName', 'pass_recipient'])['total_xt'].sum().sort_values(ascending=False).reset_index().iloc[0]
highest_xt_player_home = highest_xt_row_home['playerName']
highest_xt_recipient_home = highest_xt_row_home['pass_recipient']


highest_xt_combo_home = f"{highest_xt_player_home} -> {highest_xt_recipient_home}"


home_name = home['team_name'].iloc[0]
away_name = away['team_name'].iloc[0]







fig = plt.figure(figsize=(14, 7), constrained_layout=True, dpi=300)
gs = fig.add_gridspec(ncols=3, nrows=2)
fig.set_facecolor("#201D1D")
# create subplots using gridspec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])


axes = [ax1, ax2, ax3, ax4,ax5, ax6]

# apply modifications to all subplots
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


whovis.plot_pass_map_fulltime_subs_xT(ax1,data=data,teamId=opta_home_teamID,minute_start=0,minute_end=99,passes=3,touches=20,min_size=80,max_size=280,cmap_name='passnetwork')

whovis.plot_pass_map_fulltime_subs_xT_away(ax3,data=data,teamId=opta_away_teamID,minute_start=0,minute_end=99,passes=3,touches=20,min_size=80,max_size=200,cmap_name='passnetwork')



fmvis.plot_match_shotmap(ax2,Fotmob_matchID,homecolor=homecolor,awaycolor=awaycolor)
whovis.plot_xT_flow_chart(ax4, data=data,homecolor=homecolor , awaycolor=awaycolor)
whovis.plot_zone_dominance(ax5, opta_matchID, home_name,homecolor=homecolor, awaycolor=awaycolor,zonecolor='#c2c1c2', data= data)

fmvis.plot_match_xgflow(ax6, Fotmob_matchID,homecolor=homecolor,awaycolor=awaycolor)


# cmap = plt.cm.get_cmap('passnetwork')
cmap = plt.colormaps['passnetwork']


sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data['xThreat_gen'].min(), vmax=data['xThreat_gen'].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, fraction=0.04, pad=0.02, shrink=0.4, aspect=8, orientation='horizontal',anchor=(.85, 0.0))
cbar.ax.tick_params(labelsize=4, length=2, color='w')
cbar.ax.yaxis.set_ticks_position('none')
cbar.ax.xaxis.set_ticks_position('none')
# cbar.ax.set_title('Low          xT          High', loc='center', color='w', fontsize=6, fontweight='bold')
cbar.ax.annotate('Low        xT Value       High', xy=(0.5, -0.5), xytext=(0, -1.5), xycoords='axes fraction', textcoords='offset points', color='w', fontsize=6, fontweight='bold', ha='center', va='top')
cbar.outline.set_edgecolor('#201D1D')




sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data['xThreat_gen'].min(), vmax=data['xThreat_gen'].max()))
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax3, fraction=0.04, pad=0.02, shrink=0.4, aspect=8, orientation='horizontal',anchor=(.15, 0.0))
cbar.ax.tick_params(labelsize=4, length=2, color='w')
cbar.ax.yaxis.set_ticks_position('none')
cbar.ax.xaxis.set_ticks_position('none')
# cbar.ax.set_title('Low          xT          High', loc='center', color='w', fontsize=6, fontweight='bold')
cbar.ax.annotate('Low        xT Value       High', xy=(0.5, -0.5), xytext=(0, -1.5), xycoords='axes fraction', textcoords='offset points', color='w', fontsize=6, fontweight='bold', ha='center', va='top')
cbar.outline.set_edgecolor('#201D1D')




ax5.set_facecolor("#201D1D")
ax5.grid(False)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.spines['left'].set_visible(False)


team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')






away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')


away_logo_ax = fig.add_axes([.90, 1, .1, .14], frameon=False)

away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')

away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])


logo_ax = fig.add_axes([0, 1, 0.1, 0.14], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])

logo_ax.axis('off')



ax1.set_title(f'{home_name} Passing Network with xT', size="8", c="#FCE6E6", loc="center")
ax2.set_title('Shot Map', size="8", c="#FCE6E6", loc="center")
ax3.set_title(f'{away_name} Passing Network with xT', size="8", c="#FCE6E6", loc="center")
ax4.set_title('xThreat Flow Chart', size="8", c="#FCE6E6", loc="center")
ax5.set_title('Zone Dominance', size="8", c="#FCE6E6", loc="center")
ax6.set_title('xG Flow Chart', size="8", c="#FCE6E6", loc="center")



ax = fig.add_axes([.45, 1, .1, .22], frameon=False)

ax_text(
    0.5,
    0.7,
    match_score,
    fontsize=16,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.44,
    match_name,
    fontsize=12,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.2,
    match_label + " | " + match_season + " | " + matchDate,
    fontsize=8,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)
ax.set_axis_off()


from io import BytesIO

# Create a BytesIO object to store the figure
figure_buffer = BytesIO()

# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)

# Reset the buffer position to the beginning
figure_buffer.seek(0)

# Initialize Google Cloud Storage client and get the bucket
storage_client = storage.Client()
bucket_name = "postmatch-dashboards"
bucket = storage_client.get_bucket(bucket_name)

# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboardmain{today}.png"

# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")

# Close the BytesIO buffer
figure_buffer.close()





period1_end_minute = int(data[data.period == 1]['minute'].max())
period2_start_minute = int(data[data.period == 2]['minute'].min())
match_end_minute = int(data['minute'].max())
minutes = [(0, 15), (15, 30), (30, period1_end_minute),
           (period1_end_minute, period1_end_minute+15),
           (period1_end_minute+15, match_end_minute-15),
           (match_end_minute-15, match_end_minute),
           (0,match_end_minute)]


# Create the figure and gridspec
fig = plt.figure(figsize=(11, 10), constrained_layout=True, dpi=200)
gs = fig.add_gridspec(ncols=3, nrows=3)
fig.set_facecolor("#201D1D")

# Create subplots using gridspec
axes = []
for i, (start_minute, end_minute) in enumerate(minutes):
    row = i // 3
    col = i % 3
    ax = fig.add_subplot(gs[row, col])
    if start_minute == 0 and end_minute == match_end_minute:
        # plot_pass_map_fulltime_subs_xT(ax,data=data,teamId=26,minute_start=start_minute, minute_end=end_minute,passes=4,cmap_name='coolwarm')
        whovis.plot_pass_map_fulltime_subs_xT(ax,data=data,teamId=opta_home_teamID,minute_start=start_minute,minute_end=end_minute,passes=4,touches=20,min_size=40,max_size=280,cmap_name='passnetwork')
    else:
        whovis.plot_pass_map_minute_xT_grid(ax, data=data,teamId=opta_home_teamID, minute_start=start_minute, minute_end=end_minute, passes=1, touches=6, cmap_name='passnetwork')
    axes.append(ax)
    ax.set_title(f"{start_minute}-{end_minute} minutes", c='w',fontsize=7)
    cmap = plt.cm.get_cmap('passnetwork')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data['xThreat_gen'].min(), vmax=data['xThreat_gen'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.4, aspect=8, anchor=(.85, 0.0),orientation='horizontal',)
    cbar.ax.tick_params(labelsize=4, length=2, color='w')
    cbar.ax.yaxis.set_ticks_position('none')
    cbar.ax.xaxis.set_ticks_position('none')
    cbar.ax.annotate('Low        xT Value       High', xy=(0.5, -0.5), xytext=(0, -1.5), xycoords='axes fraction', textcoords='offset points', color='w', fontsize=6, fontweight='bold', ha='center', va='top')
    cbar.outline.set_edgecolor('#201D1D')


team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')

logo_ax = fig.add_axes([0, 1, 0.10, 0.08], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])

logo_ax.axis('off')

bbox_props = dict(boxstyle='round,pad=0.6',facecolor=homecolor, alpha=.4, edgecolor="k")

fig.suptitle('Passing Network', fontsize=10, fontweight='bold', y=1.01, x=0.5,c='#FCE6E6' ,  ha='center',va='top')

fig.text(0.4, 0.26, "Most Passes", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.4, 0.22, passes_home, color='w', fontsize=8)
fig.text(0.80, 0.26, "Most xT via Passes", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.80, 0.22, xT_pass_home, color='w', fontsize=8)

fig.text(0.6, 0.26, "Most Prog Passes Rec", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.6, 0.22, prog_rec_home, color='w', fontsize=8)


fig.text(0.60, 0.18, "Most xT Rec", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.60, 0.14, xT_rec_home, color='w', fontsize=8)
fig.text(0.4, 0.18, "Most Prog Passes", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.4, 0.14, prog_passes_home, color='w', fontsize=8)
fig.text(0.8, 0.18, "Most Passes Rec", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.8, 0.14, pass_rec_home, color='w', fontsize=8)


fig.text(0.4, 0.08, "Highest xT Combo", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.4, 0.04, highest_xt_combo_home, color='w', fontsize=8)

fig.text(0.60, 0.08, "Most Passes Between", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.6, 0.04, most_pass_combo_home, color='w', fontsize=8)



fig.text(0.8, 0.08, "Most Prog Passes Between", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.8, 0.04, most_pass_combo_home, color='w', fontsize=8)




# plt.savefig(
#     f"Post_Match_Dashboard/figures/dashboardpassnetworkhome{matchDate}.png",
#     dpi = 700,
#     bbox_inches="tight",
#     edgecolor="none",
#     transparent = False
# )

# Create a BytesIO object to store the figure
figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboardpassnetworkhome{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()



fig = plt.figure(figsize=(11, 10), constrained_layout=True, dpi=300)
gs = fig.add_gridspec(ncols=3, nrows=3)
fig.set_facecolor("#201D1D")

# Create subplots using gridspec
axes = []
for i, (start_minute, end_minute) in enumerate(minutes):
    row = i % 3
    col = i // 3
    ax = fig.add_subplot(gs[col, 2-row])  # Change the row and column indices
    # Plot the pass map on the rotated subplot
    if start_minute == 0 and end_minute == match_end_minute:
        whovis.plot_pass_map_fulltime_subs_xT_away(ax,data=data,teamId=opta_away_teamID,minute_start=start_minute,minute_end=end_minute,passes=4,touches=20,min_size=40,max_size=180,cmap_name='passnetwork')

    else:
        whovis.plot_pass_map_minute_xT_away_grid(ax, data=data, teamId=opta_away_teamID, minute_start=start_minute, minute_end=end_minute, passes=1, touches=6,min_size=40,max_size=180, cmap_name='passnetwork')
    axes.append(ax)
    ax.set_title(f"{start_minute}-{end_minute} minutes", c='w', fontsize=6)
    cmap = plt.cm.get_cmap('passnetwork')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data['xThreat_gen'].min(), vmax=data['xThreat_gen'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.4, aspect=8, anchor=(.75, 0.0),orientation='horizontal',)
    cbar.ax.tick_params(labelsize=4, length=2, color='w')
    cbar.ax.yaxis.set_ticks_position('none')
    cbar.ax.xaxis.set_ticks_position('none')
    cbar.ax.annotate('Low        xT Value       High', xy=(0.5, -0.5), xytext=(0, -1.5), xycoords='axes fraction', textcoords='offset points', color='w', fontsize=6, fontweight='bold', ha='center', va='top')
    cbar.outline.set_edgecolor('#201D1D')

bbox_props = dict(boxstyle='round,pad=0.6',facecolor=awaycolor, alpha=.4, edgecolor="#51B198")



fig.text(0.2, 0.26, "Most Passes", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.2, 0.22, passes_away, color='w', fontsize=8)
fig.text(0.35, 0.26, "Most xT via Passes", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.35, 0.22, xT_pass_away, color='w', fontsize=8)
fig.text(0.5, 0.26, "Most Prog Passes Rec", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.5, 0.22, prog_rec_away, color='w', fontsize=8)
fig.text(0.35, 0.18, "Most xT Rec", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.35, 0.14, xT_rec_away, color='w', fontsize=8)
fig.text(0.2, 0.18, "Most Prog Passes", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.2, 0.14, prog_passes_away, color='w', fontsize=8)
fig.text(0.5, 0.18, "Most Passes Rec", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.5, 0.14, pass_rec_away, color='w', fontsize=8)
fig.text(0.2, 0.08, "Highest xT Combo", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.2, 0.04, highest_xt_combo_away, color='w', fontsize=8)
fig.text(0.35, 0.08, "Most Passes Between", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.35, 0.04, most_pass_combo_away, color='w', fontsize=8)
fig.text(0.5, 0.08, "Most Prog Passes Between", color='w', fontsize=8, bbox=bbox_props)
fig.text(0.5, 0.04, most_pass_combo_away, color='w', fontsize=8)




away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')

#away_logo_ax = ax.inset_axes([0.75, 0.8, 0.2, 0.2], transform=ax.transAxes)
# away_logo_ax = fig.add_axes([0.06, 0.1, 0.12, 0.18], frameon=False)
away_logo_ax = fig.add_axes([.90, 1, .1, .08], frameon=False)


away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')

away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])




fig.suptitle('Passing Network', fontsize=10, fontweight='bold', y=1.01, x=0.5,c='#FCE6E6' ,  ha='center',va='top')


# plt.savefig(
#     f"Post_Match_Dashboard/figures/dashboardpassnetworkaway{matchDate}.png",
#     dpi = 600,
#     bbox_inches="tight",
#     edgecolor="none",
#     transparent = False
# )

# Create a BytesIO object to store the figure
figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboardpassnetworkaway{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()


def find_offensive_actions(events_df):
    # Define and filter offensive events
    offensive_actions = ['TakeOn', 'MissedShots', 'SavedShot', 'Goal','ShotOnPost', 'Carry','Pass']
    offensive_action_df = events_df[events_df['event_type'].isin(offensive_actions)].reset_index(drop=True)

    # Filter for passes made by goalkeepers
    pass_gk = events_df[(events_df['event_type'] == 'Pass') & (events_df['position'] == 'GK')]

    # Filter for passes with assists or pre-assists
    # pass_df = events_df[(events_df['is_progressive'] == True) & ((events_df['assist'] == True) | (events_df['pre_assist'] == True))]
    pass_df = events_df[(events_df['event_type'] == 'Pass') & (events_df['is_progressive']) & (events_df['assist'] | events_df['pre_assist'])]

    # Concatenate offensive actions and passes with assists or pre-assists
    offensive_actions_df = pd.concat([offensive_action_df, pass_df, pass_gk]).reset_index(drop=True)

    return offensive_actions_df

def find_defensive_actions(events_df):
    """ Return dataframe of in-play defensive actions from event data.
    Function to find all in-play defensive actions within a whscored-style events dataframe (single or multiple
    matches), and return as a new dataframe.
    Args:
        events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
    Returns:
        pandas.DataFrame: whoscored-style dataframe of defensive actions.
    """

    # Define and filter defensive events
    defensive_actions = ['BallRecovery', 'BlockedPass', 'Challenge', 'Clearance', 'Interception', 'Tackle',
                         'Claim', 'KeeperPickup', 'Punch', 'Save','BlockedPass','Aerial','Dispossessed']
    defensive_action_df = events_df[events_df['event_type'].isin(defensive_actions)]

    return defensive_action_df

offensive_actions = find_offensive_actions(data)
defensive_actions = find_defensive_actions(data)

fig = plt.figure(figsize=(12, 12), constrained_layout=True, dpi=100)
gs = fig.add_gridspec(ncols=2, nrows=2)
fig.set_facecolor("#201D1D")
# create subplots using gridspec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

axes = [ax1, ax2, ax3, ax4]

# apply modifications to all subplots
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



whovis.plot_team_offensive_actions(ax1, opta_matchID, opta_home_teamID, offensive_actions,color=homecolor)


whovis.plot_team_offensive_actions(ax2, opta_matchID, opta_away_teamID, offensive_actions,color=awaycolor)

whovis.plot_team_defensive_actions_opp_half(ax4, opta_matchID, opta_away_teamID, defensive_actions,color=awaycolor)

whovis.plot_team_defensive_actions_opp_half(ax3, opta_matchID, opta_home_teamID, defensive_actions,color=homecolor)


# add titles to the subplots
ax1.set_title(f"{home_name} Offensive Actions",size="8", c="#FCE6E6", loc="center")
ax2.set_title(f"{away_name} Offensive Actions",size="8", c="#FCE6E6", loc="center")
ax3.set_title(f"{home_name} Defensive Actions in Opponent Half",size="8", c="#FCE6E6", loc="center")
ax4.set_title(f"{away_name} Defensive Actions in Opponent Half",size="8", c="#FCE6E6", loc="center")



team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')




away_logo_ax = fig.add_axes([.86, .98, .08, .08], frameon=False)
away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')
away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])
logo_ax = fig.add_axes([0.06, .98, 0.08, 0.08], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])
logo_ax.axis('off')



# create a new Axes object on the figure
# ax = fig.add_axes([0.5, 1, 0.0, 0.0], frameon=False)
ax = fig.add_axes([.45, 1, .1, .10], frameon=False)

# add the text to the Axes object using ax_text
ax_text(
    0.5,
    0.7,
    match_score,
    fontsize=16,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.44,
    match_name,
    fontsize=12,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.2,
    match_label + " | " + match_season + " | " + matchDate,
    fontsize=8,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

# finally, turn off the visibility of the Axes object
ax.set_axis_off()




# fig.savefig(f"Post_Match_Dashboard/figures/dashboard_teamactions{matchDate}.png", dpi=600, bbox_inches="tight")


figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboard_teamactions{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()

fig = plt.figure(figsize=(12, 10), constrained_layout=True, dpi=650)
gs = fig.add_gridspec(ncols=2, nrows=2)
fig.set_facecolor("#201D1D")
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

axes = [ax1, ax2,ax3,ax4]

# apply modifications to all subplots
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


whovis.plot_clusters_event(ax1,data=data,teamId=opta_home_teamID,k=6,filter="is_progressive == True & is_open_play == True")
whovis.plot_clusters_event(ax3,data=data,teamId=opta_home_teamID,k=8,filter="is_open_play == True")
whovis.plot_event_clusters_away(ax2,data=data,teamId=opta_away_teamID,k=6,filter="is_progressive == True & is_open_play == True")
whovis.plot_event_clusters_away(ax4,data=data,teamId=opta_away_teamID,k=8,filter="is_open_play == True")


# add titles to the subplots
ax1.set_title(f"{home_name} Progressive Passes Cluster",size="10", c="#FCE6E6", loc="center")
ax2.set_title(f"{away_name} Progressive Passes Cluster",size="10", c="#FCE6E6", loc="center")

ax3.set_title(f"{home_name} All Attempted Passes Cluster",size="10", c="#FCE6E6", loc="center")
ax4.set_title(f"{away_name} All Attempted Passes Cluster",size="10", c="#FCE6E6", loc="center")

team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')

#away_logo_ax = ax.inset_axes([0.75, 0.8, 0.2, 0.2], transform=ax.transAxes)
away_logo_ax = fig.add_axes([.86, 1, .14, .08], frameon=False)


away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')

away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])


logo_ax = fig.add_axes([.02, 1, 0.14, 0.08], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])


#logo_ax = fig.add_axes([0, 1, 0.1, 0.1], frameon=False)
#logo_ax.imshow(club_icon)
logo_ax.axis('off')

# create a new Axes object on the figure
# ax = fig.add_axes([0.5, 1, 0.0, 0.0], frameon=False)


ax = fig.add_axes([.45, 1, .1, .10], frameon=False)

# add the text to the Axes object using ax_text
ax_text(
    0.5,
    0.7,
    match_score,
    fontsize=26,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.44,
    match_name,
    fontsize=14,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.2,
    match_label + " | " + match_season + " | " + matchDate,
    fontsize=12,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

# finally, turn off the visibility of the Axes object
ax.set_axis_off()





# fig.savefig(f"Post_Match_Dashboard/figures/dashboard_cluster_passes{matchDate}.png", dpi=540, bbox_inches="tight")


figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=600,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboard_cluster_passes{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()


fig = plt.figure(figsize=(12, 8), constrained_layout=True, dpi=300)
gs = fig.add_gridspec(ncols=2, nrows=2)
fig.set_facecolor("#201D1D")
# create subplots using gridspec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

axes = [ax1, ax2, ax3, ax4]

# apply modifications to all subplots
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



whovis.plot_xT_heatmap(ax1,data=data,teamId=opta_home_teamID,color=homecolor)


whovis.plot_xT_heatmap_away(ax2,data=data,teamId=opta_away_teamID,color=awaycolor)


whovis.plot_deep_comp_team(ax3,data=data,teamId=opta_home_teamID,passColor='#8fb5ab',carryColor='#fa7d3b',radius=26)
whovis.plot_deep_comp_team(ax4,data=data,teamId=opta_away_teamID,passColor='#8fb5ab',carryColor='#fa7d3b',radius=26)

ax1.set_title(f"{home_name} xThreat Heatmap ", size="8", c="#FCE6E6", loc="center")
ax2.set_title(f"{away_name} xThreat Heatmap", size="8", c="#FCE6E6", loc="center")
ax3.set_title(f"{home_name} Deep Completion", size="8", c="#FCE6E6", loc="center")
ax4.set_title(f"{away_name} Deep Completion ",size="8", c="#FCE6E6", loc="center")



team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')


#away_logo_ax = ax.inset_axes([0.75, 0.8, 0.2, 0.2], transform=ax.transAxes)
away_logo_ax = fig.add_axes([.90, .98, .14, .08], frameon=False)


away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')

away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])


logo_ax = fig.add_axes([0, .98, 0.14, 0.08], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])


#logo_ax = fig.add_axes([0, 1, 0.1, 0.1], frameon=False)
#logo_ax.imshow(club_icon)
logo_ax.axis('off')
# create a new Axes object on the figure
# ax = fig.add_axes([0.5, 1, 0.0, 0.0], frameon=False)
ax = fig.add_axes([.45, 1, .1, .14], frameon=False)

# add the text to the Axes object using ax_text
ax_text(
    0.5,
    0.7,
    match_score,
    fontsize=26,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.44,
    match_name,
    fontsize=14,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.2,
    match_label + " | " + match_season + " | " + matchDate,
    fontsize=12,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

# finally, turn off the visibility of the Axes object
ax.set_axis_off()


# fig.savefig(f"Post_Match_Dashboard/figures/dashboard_xT_heatmap{matchDate}.png", dpi=700, bbox_inches="tight")


figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboard_xT_heatmap{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()






fig = plt.figure(figsize=(12, 8), constrained_layout=True, dpi=300)
gs = fig.add_gridspec(ncols=2, nrows=2)
fig.set_facecolor("#201D1D")
# create subplots using gridspec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

axes = [ax1, ax2, ax3, ax4]

# apply modifications to all subplots
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



whovis.plot_team_passes_filter(ax=ax1,data=data,teamId=opta_home_teamID,color=homecolor,filter="FinalThirdPasses == True ")

whovis.plot_team_passes_filter(ax3,data=data,teamId=opta_home_teamID,color=homecolor,filter="is_pass_into_box == True ")
whovis.plot_team_passes_filter_away(ax2,data=data,teamId=opta_away_teamID,color=awaycolor,filter="FinalThirdPasses == True")
whovis.plot_team_passes_filter_away(ax4,data=data,teamId=opta_away_teamID,color=awaycolor,filter="is_pass_into_box == True")

ax1.set_title(f"{home_name} Final Third Passes  ", size="8", c="#FCE6E6", loc="center")
ax2.set_title(f"{away_name} Final Third Passes", size="8", c="#FCE6E6", loc="center")
ax3.set_title(f"{home_name} Passes Into The Box", size="8", c="#FCE6E6", loc="center")
ax4.set_title(f"{away_name} Passes Into The Box ",size="8", c="#FCE6E6", loc="center")



team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')


#away_logo_ax = ax.inset_axes([0.75, 0.8, 0.2, 0.2], transform=ax.transAxes)
away_logo_ax = fig.add_axes([.90, .98, .14, .08], frameon=False)


away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')

away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])


logo_ax = fig.add_axes([0, .98, 0.14, 0.08], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])


#logo_ax = fig.add_axes([0, 1, 0.1, 0.1], frameon=False)
#logo_ax.imshow(club_icon)
logo_ax.axis('off')
# create a new Axes object on the figure
# ax = fig.add_axes([0.5, 1, 0.0, 0.0], frameon=False)
ax = fig.add_axes([.45, 1, .1, .14], frameon=False)

# add the text to the Axes object using ax_text
ax_text(
    0.5,
    0.7,
    match_score,
    fontsize=26,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.44,
    match_name,
    fontsize=14,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.2,
    match_label + " | " + match_season + " | " + matchDate,
    fontsize=12,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

# finally, turn off the visibility of the Axes object
ax.set_axis_off()



figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboard_passes{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()




fig = plt.figure(figsize=(12, 14), constrained_layout=True, dpi=650)
gs = fig.add_gridspec(ncols=2, nrows=2)
fig.set_facecolor("#201D1D")
# create subplots using gridspec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

axes = [ax1, ax2]

whovis.plot_team_heatmap(ax1,data=data,teamid=opta_home_teamID,color=homecolor,sd=0)

whovis.plot_team_heatmap_away(ax2,data=data,teamid=opta_away_teamID,color=awaycolor,sd=0)



team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')


#away_logo_ax = ax.inset_axes([0.75, 0.8, 0.2, 0.2], transform=ax.transAxes)
away_logo_ax = fig.add_axes([.90, .96, .10, .08], frameon=False)


away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')

away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])


logo_ax = fig.add_axes([0, .96, 0.10, 0.08], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])


#logo_ax = fig.add_axes([0, 1, 0.1, 0.1], frameon=False)
#logo_ax.imshow(club_icon)
logo_ax.axis('off')
# create a new Axes object on the figure
# ax = fig.add_axes([0.5, 1, 0.0, 0.0], frameon=False)


fig_text(
    0.5,
    0.98,
    match_score,
    fontsize=9,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

fig_text(
    0.5,
    0.96,
    match_name,
    fontsize=9,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

fig_text(
    0.5,
    0.94,
    match_label + " | " + match_season + " | " + matchDate,
    fontsize=7,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

# finally, turn off the visibility of the Axes object
ax.set_axis_off()



ax1.set_title(f"{home_name} Touch Heatmap ", size="8", c="#FCE6E6", loc="center")
ax2.set_title(f"{away_name} Touch Heatmap", size="8", c="#FCE6E6", loc="center")

# fig.savefig(f"Post_Match_Dashboard/figures/dashboard_Touchheatmap{matchDate}.png", dpi=700, bbox_inches="tight")


figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboard_Touchheatmap{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()



fig = plt.figure(figsize=(10, 8), constrained_layout=True, dpi=650)
gs = fig.add_gridspec(ncols=2, nrows=2)
fig.set_facecolor("#201D1D")
# create subplots using gridspec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

axes = [ax1, ax2]

whovis.plot_team_turnovers(ax1,data=data,teamId=opta_home_teamID,c=homecolor)

whovis.plot_team_turnovers(ax2,data=data,teamId=opta_away_teamID,c=awaycolor)



team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')


#away_logo_ax = ax.inset_axes([0.75, 0.8, 0.2, 0.2], transform=ax.transAxes)
away_logo_ax = fig.add_axes([.90, .96, .10, .08], frameon=False)


away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')

away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])


logo_ax = fig.add_axes([0, .96, 0.10, 0.08], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])


#logo_ax = fig.add_axes([0, 1, 0.1, 0.1], frameon=False)
#logo_ax.imshow(club_icon)
logo_ax.axis('off')
# create a new Axes object on the figure
# ax = fig.add_axes([0.5, 1, 0.0, 0.0], frameon=False)


fig_text(
    0.5,
    0.98,
    match_score,
    fontsize=9,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

fig_text(
    0.5,
    0.96,
    match_name,
    fontsize=9,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

fig_text(
    0.5,
    0.94,
    match_label + " | " + match_season + " | " + matchDate,
    fontsize=7,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

# finally, turn off the visibility of the Axes object
ax.set_axis_off()



ax1.set_title(f"{home_name} Turnovers", size="8", c="#FCE6E6", loc="center")
ax2.set_title(f"{away_name} Turnovers", size="8", c="#FCE6E6", loc="center")

# fig.savefig(f"Post_Match_Dashboard/figures/dashboard_Touchheatmap{matchDate}.png", dpi=700, bbox_inches="tight")


figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboard_turnovers{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()




#%%

#%%

#%%

fig = plt.figure(figsize=(12, 6), constrained_layout=True, dpi=700)
gs = fig.add_gridspec(ncols=2, nrows=1)
fig.set_facecolor("#201D1D")

# create subplots using gridspec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

axes = [ax1, ax2]
# apply modifications to all subplots
for ax in axes:
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor("#201D1D")



whovis.plot_xT_players_barplot(ax1,data=data, teamid=opta_home_teamID, event_type='Pass', color=homecolor)

whovis.plot_xT_players_barplot(ax2,data=data, teamid=opta_away_teamID, event_type='Pass', color=awaycolor)

# whovis.plot_xT_players_barplot(ax3,data=data, teamid=opta_home_teamID, event_type='Carry', color='#AC944D')

# whovis.plot_xT_players_barplot(ax4,data=data, teamid=opta_away_teamID, event_type='Carry', color='#dfab27')

# colors_list = ['#2a9d8f', '#dfab27']
# colors_list = homecolor + awaycolor

# whovis.plot_xT_rec_barplot_teams(ax2,data=data, event_type='Pass', colors=colors_list)


ax1.set_title(f"{home_name} Players xT via Passes", size="12", c="#E1D3D3", loc="center")
ax2.set_title(f"{away_name} Players xT via Passes", size="12", c="#E1D3D3", loc="center")
# ax2.set_title("Both Teams Players xT Received", size="12", c="#E1D3D3", loc="center")



# fig.savefig(f"Post_Match_Dashboard/figures/dashboard_xT_bar{matchDate}.png", dpi=700, bbox_inches="tight")



figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboard_xT_bar{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()


# homecolor = ['#2a9d8f']
# awaycolor = ['#dfab27']
# colors_list = homecolor + awaycolor


shots_data = shots_data.rename(columns={'expectedGoals':'xG'})
shots_data = shots_data[shots_data['isOwnGoal']==0]

fig = plt.figure(figsize=(14, 8), constrained_layout=True, dpi=100)
gs = fig.add_gridspec(ncols=3, nrows=2)
fig.set_facecolor("#201D1D")
# create subplots using gridspec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])


axes = [ax1, ax2, ax3, ax4,ax5, ax6]

# apply modifications to all subplots
for ax in axes:
    # ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor("#201D1D")





odds.plot_players_xg_dis(ax1,Fotmob_homeID, color=homecolor,data=shots_data)
odds.plot_players_xg_dis(ax3, Fotmob_awayID, color=awaycolor,data=shots_data)

odds.plot_goals_prob(ax4, data=shots_data, team_id=Fotmob_homeID, color= homecolor)

odds.plot_goals_prob(ax6,data=shots_data,team_id= Fotmob_awayID,color=awaycolor)


odds.simulate_and_plot_match_result(ax2,colors=[homecolor, '#8a9e9d', awaycolor],shot_df=shots_data, k=10000)

odds.plot_score_probability_matrix(ax5, cmap='passnetwork',data=shots_data)




ax1.set_title(f"{home_name} Players Expected Goals Distribution", size="8", c="#FCE6E6", loc="center")
ax2.set_title("Match Result Simulation", size="8", c="#FCE6E6", loc="center")
ax3.set_title(f"{away_name} Players Expected Goals Distribution", size="8", c="#FCE6E6", loc="center")
ax4.set_title(f"{home_name} Goal Probability",size="8", c="#FCE6E6", loc="center")
ax5.set_title("Score Probability Matrix", size="8", c="#FCE6E6", loc="center")
ax6.set_title(f"{away_name}Goal Probability",size="8", c="#FCE6E6", loc="center")

team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')

away_logo_ax = fig.add_axes([.90, 1, .1, .10], frameon=False)


away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')

away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])


logo_ax = fig.add_axes([0, 1, 0.1, 0.10], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])

# create a new Axes object on the figure
# ax = fig.add_axes([0.5, 1, 0.0, 0.0], frameon=False)
ax = fig.add_axes([.45, 1, .1, .14], frameon=False)

# add the text to the Axes object using ax_text
ax_text(
    0.5,
    0.7,
    match_score,
    fontsize=26,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.44,
    match_name,
    fontsize=14,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

ax_text(
    0.5,
    0.2,
    match_label + " | " + match_season + " | " + matchDate,
    fontsize=12,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

# finally, turn off the visibility of the Axes object
ax.set_axis_off()
away_logo_ax.axis('off')
logo_ax.axis('off')


# fig.savefig(f"Post_Match_Dashboard/figures/dashboard_odds{matchDate}.png", dpi=600, bbox_inches="tight")

figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboard_odds{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()





def get_player_dfs(data):
    player_dfs = {}

    for player_name in data['playerName'].dropna().unique():
        player_df = data[data['playerName'] == player_name].copy()

        x_avg = player_df['x'].mean()
        y_avg = player_df['y'].mean()

        player_df['x_avg'] = x_avg
        player_df['y_avg'] = y_avg

        initials = player_name.split()[0][0] + player_name.split()[-1][0]
        player_df['initials'] = initials

        player_dfs[player_name] = player_df

    return player_dfs


player_offensive_actions = get_player_dfs(offensive_actions)

player_dfs = get_player_dfs(defensive_actions)




Home_gk = data[(data['Venue'] == 'Home') & (data['position'].isin(['GK']))]['playerName'].unique()
Home_cb = data[(data['Venue'] == 'Home') & (data['position'].isin(['DC']))]['playerName'].unique()
Home_fullbacks = data[(data['Venue'] == 'Home') & (data['position'].isin(['DL','DR']))]['playerName'].unique()
Home_midfielders = data[(data['Venue'] == 'Home') & (data['position'].isin(['DMC', 'AMC', 'MC']))]['playerName'].unique()
Home_attackers = data[(data['Venue'] == 'Home') & (data['position'].isin(['FWR', 'FWL', 'AML','AMR','FW']))]['playerName'].unique()
Home_sub = data[(data['Venue'] == 'Home') & (data['position'].isin(['Sub']))]['playerName'].unique()


Away_gk = data[(data['Venue'] == 'Away') & (data['position'].isin(['GK']))]['playerName'].unique()
Away_cb = data[(data['Venue'] == 'Away') & (data['position'].isin(['DC']))]['playerName'].unique()
Away_fullbacks = data[(data['Venue'] == 'Away') & (data['position'].isin(['DL','DR']))]['playerName'].unique()
Away_midfielders = data[(data['Venue'] == 'Away') & (data['position'].isin(['DMC', 'AMC', 'MC']))]['playerName'].unique()
Away_attackers = data[(data['Venue'] == 'Away') & (data['position'].isin(['FWR', 'FWL', 'AML','AMR','FW']))]['playerName'].unique()
Away_sub = data[(data['Venue'] == 'Away') & (data['position'].isin(['Sub']))]['playerName'].unique()



fig = plt.figure(figsize=(8, 6), constrained_layout=True, dpi=300)
gs = fig.add_gridspec(ncols=2, nrows=2)
fig.set_facecolor("#201D1D")
# create subplots using gridspec
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

axes = [ax1, ax2, ax3, ax4]

# apply modifications to all subplots
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
    pitch = Pitch(
        pitch_color='#201D1D',
        pitch_type='opta',
        goal_type='box',
        linewidth=1.25,
        line_color='black',
        half=False
    )
    pitch.draw(ax = ax)

    pos_x = pitch.dim.positional_x
    pos_y = pitch.dim.positional_y
    for x in pos_x[1:-1]:
        ax.plot([x,x], [pos_y[0], pos_y[-1]], color='#000000', ls='dashed', zorder=0, lw=0.3)
    for y in pos_y[1:-1]:
        ax.plot([pos_x[0], pos_x[-1]], [y,y], color='#000000', ls='dashed', zorder=0, lw=0.3)




ax1.set_title(f"{home_name} Territory Map Defensive Actions",size="6", c="#EFE9F4", loc="center")
ax2.set_title(f"{away_name} Territory Map Defensive Actions",size="6", c="#EFE9F4", loc="center")

ax3.set_title(f"{home_name} Territory Map Offensive Actions",size="6", c="#EFE9F4", loc="center")
ax4.set_title(f"{away_name} Territory Map Offensive Actions",size="6", c="#EFE9F4", loc="center")



team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_homeID}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


away_team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{Fotmob_awayID}.png'
away_club_icon = Image.open(away_team_logo_path).convert('RGBA')



for player in Home_gk:
    player_df = player_dfs[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax1, poly_edgecolor='#44335a', poly_facecolor='#44335a',
                                     scatter_facecolor='#44335a', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Home_gk:
    player_df = player_offensive_actions[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax3, poly_edgecolor='#44335a', poly_facecolor='#44335a',
                                     scatter_facecolor='#44335a', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Home_cb:
    player_df = player_dfs[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax1, poly_edgecolor='#674997', poly_facecolor='#674997',
                                     scatter_facecolor='#674997', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Home_cb:
    player_df = player_offensive_actions[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax3, poly_edgecolor='#674997', poly_facecolor='#674997',
                                     scatter_facecolor='#674997', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Home_fullbacks:
    player_df = player_dfs[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax1, poly_edgecolor='#007a72', poly_facecolor='#007a72',
                                     scatter_facecolor='#007a72', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Home_fullbacks:
    player_df = player_offensive_actions[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax3, poly_edgecolor='#007a72', poly_facecolor='#007a72',
                                     scatter_facecolor='#007a72', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Home_midfielders:
    player_df = player_dfs[player]
    if not player_df.empty and len(player_df) > 3:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax1, poly_edgecolor='#d1495b', poly_facecolor='#d1495b',
                                     scatter_facecolor='#d1495b', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")



for player in Home_midfielders:
    player_df = player_offensive_actions[player]
    if not player_df.empty and len(player_df) > 3:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax3, poly_edgecolor='#d1495b', poly_facecolor='#d1495b',
                                     scatter_facecolor='#d1495b', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Home_attackers:
    player_df = player_dfs[player]
    if not player_df.empty and len(player_df) > 3:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax1, poly_edgecolor='#edae49', poly_facecolor='#edae49',
                                     scatter_facecolor='#edae49', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Home_attackers:
    player_df = player_offensive_actions[player]
    if not player_df.empty and len(player_df) > 3:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_team(player_df, ax3, poly_edgecolor='#edae49', poly_facecolor='#edae49',
                                     scatter_facecolor='#edae49', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")


for player in Away_gk:
    player_df = player_dfs[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax2, poly_edgecolor='#44335a', poly_facecolor='#44335a',
                                     scatter_facecolor='#44335a', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Away_gk:
    player_df = player_offensive_actions[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax4, poly_edgecolor='#44335a', poly_facecolor='#44335a',
                                     scatter_facecolor='#44335a', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Away_cb:
    player_df = player_dfs[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax2, poly_edgecolor='#674997', poly_facecolor='#674997',
                                     scatter_facecolor='#674997', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Away_cb:
    player_df = player_offensive_actions[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax4, poly_edgecolor='#674997', poly_facecolor='#674997',
                                     scatter_facecolor='#674997', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Away_fullbacks:
    player_df = player_dfs[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax2, poly_edgecolor='#007a72', poly_facecolor='#007a72',
                                     scatter_facecolor='#007a72', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Away_fullbacks:
    player_df = player_offensive_actions[player]
    if not player_df.empty:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax4, poly_edgecolor='#007a72', poly_facecolor='#007a72',
                                     scatter_facecolor='#007a72', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Away_midfielders:
    player_df = player_dfs[player]
    if not player_df.empty and len(player_df) > 3:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax2, poly_edgecolor='#d1495b', poly_facecolor='#d1495b',
                                     scatter_facecolor='#d1495b', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")



for player in Away_midfielders:
    player_df = player_offensive_actions[player]
    if not player_df.empty and len(player_df) > 3:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax4, poly_edgecolor='#d1495b', poly_facecolor='#d1495b',
                                     scatter_facecolor='#d1495b', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Away_attackers:
    player_df = player_dfs[player]
    if not player_df.empty and len(player_df) > 3:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax2, poly_edgecolor='#edae49', poly_facecolor='#edae49',
                                     scatter_facecolor='#edae49', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

for player in Away_attackers:
    player_df = player_offensive_actions[player]
    if not player_df.empty and len(player_df) > 3:  # check if there are at least 5 points for the player
        whovis.plot_player_hull_awayteam(player_df, ax4, poly_edgecolor='#edae49', poly_facecolor='#edae49',
                                     scatter_facecolor='#edae49', avg_marker_size=200, sd=2)
    else:
        print(f"Not enough points for {player}. Skipping...")

away_logo_ax = fig.add_axes([.90, 1, .10, .08], frameon=False)


away_logo_ax.imshow(away_club_icon, aspect='equal')
away_logo_ax.axis('off')

away_logo_ax.set_xticks([])
away_logo_ax.set_yticks([])


logo_ax = fig.add_axes([0, 1, 0.10, 0.08], frameon=False)
logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])


#logo_ax = fig.add_axes([0, 1, 0.1, 0.1], frameon=False)
#logo_ax.imshow(club_icon)
logo_ax.axis('off')
# create a new Axes object on the figure
# ax = fig.add_axes([0.5, 1, 0.0, 0.0], frameon=False)


fig_text(
    0.5,
    1.12,
    match_score,
    fontsize=8,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

fig_text(
    0.5,
    1.06,
    match_name,
    fontsize=6,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)

fig_text(
    0.5,
    1.01,
    match_label + " | " + match_season + " | " + matchDate,
    fontsize=6,
    color="#FCE6E6",
    ha="center",
    va="center",
    transform=ax.transAxes
)




ax.set_axis_off()

# fig.savefig(f"Post_Match_Dashboard/figures/dashboardTerritory_Map{matchDate}.png",bbox_inches="tight" ,dpi=600)

figure_buffer = BytesIO()
# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=650,
    bbox_inches="tight",
    edgecolor="none",
    transparent=False
)
# Reset the buffer position to the beginning
figure_buffer.seek(0)
# Specify the blob path within the bucket
blob_path = f"figures/{today}/dashboardTerritory_Map{matchDate}.png"
# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")
# Close the BytesIO buffer
figure_buffer.close()
