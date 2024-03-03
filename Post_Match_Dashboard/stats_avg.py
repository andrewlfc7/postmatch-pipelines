#%%

import datetime
import json
import os
from google.cloud import storage
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from PIL import Image
from highlight_text import fig_text
import yaml
from sqlalchemy import create_engine
import datetime
import pytz  # Make sure to install this library if you haven't already

# Set the time zone to Eastern Time
eastern = pytz.timezone('US/Eastern')

# Get the current date in Eastern Time
today = datetime.datetime.now(eastern).date()
today = today.strftime('%Y-%m-%d')


user = os.environ['PGUSER']
passwd = os.environ['PGPASSWORD']
host = os.environ['PGHOST']
port = os.environ['PGPORT']
db = os.environ['PGDATABASE']
db_url = f'postgresql://{user}:{passwd}@{host}:{port}/{db}'

engine = create_engine(db_url)



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



current_competition = 'Premier League'

conn = engine.connect()

#%%
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)
session = Session()
#%%
session.begin()
#%%
# Find the most recent match
recent_match_query = f"""
    SELECT * FROM fotmob_shots_data WHERE match_date = '{today}'
"""

# Execute the query to get the most recent match and its competition
recent_match = pd.read_sql(recent_match_query, conn)

# Check the competition of the most recent match
most_recent_competition = recent_match['competition'].iloc[0]

# Construct the subsequent queries based on the competition of the most recent match
if most_recent_competition == 'Premier League':
    shots_query = """
        SELECT * 
        FROM fotmob_shots_data 
        WHERE competition = 'Premier League'
    """
    opta_query = """
        SELECT * 
        FROM opta_event_data 
        WHERE competition = 'Premier League'
    """
    comp_name = 'Premier League'
else:
    shots_query = """
        SELECT * 
        FROM fotmob_shots_data 
        WHERE competition IN ('Premier League', '{}')
    """.format(most_recent_competition)
    opta_query = """
        SELECT * 
        FROM opta_event_data 
        WHERE competition IN ('Premier League', '{}')
    """.format(most_recent_competition)
    comp_name = 'Premier League & {}'.format(most_recent_competition)

# Query the database and load data into DataFrames
shots_data = pd.read_sql(shots_query, conn)
opta_data = pd.read_sql(opta_query, conn)

#%%

opta_data = opta_data.rename(columns={"match_id":"matchId"})

shots_data = shots_data.rename(columns={"match_id":"matchId"})

match_date = today

Fotmob_matchID = shots_data[shots_data['match_date'] == match_date]['matchId'].iloc[0]

opta_matchID = opta_data[opta_data['match_date'] == match_date]['matchId'].iloc[0]

#%%


def calculate_match_shots_stats(data,teamId):

    data['situation'] = data['situation'].replace({
        'RegularPlay': 'RegularPlay',
        'FromCorner': 'SetPiece',
        'SetPiece': 'SetPiece',
        'FastBreak': 'RegularPlay',
        'FreeKick': 'SetPiece',
        'ThrowInSetPiece': 'SetPiece',
        'Penalty': 'Penalty'
    })


    liv_data = data[data['teamId']==teamId]


    team_matches = data[data['teamId']==teamId]['matchId'].unique()

    opponents_data = data[(data['matchId'].isin(team_matches)) & (data['teamId']!=teamId)]


    #--- Liverpool
    xG = liv_data.groupby(['matchId','teamId'])['expectedGoalsOnTarget'].sum().reset_index()
    xG = xG.rename(columns={'expectedGoalsOnTarget': 'xGOT_liv'})

    npxG = liv_data[liv_data['situation']!='Penalty'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()

    openplay_xG = liv_data[liv_data['situation']=='RegularPlay'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()
    setpiece_xG = liv_data[liv_data['situation']=='SetPiece'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()
    openplay_shots = liv_data[liv_data['situation']=='RegularPlay'].groupby(['matchId', 'teamId']).size().reset_index(name='shotCount')
    openplay_xG = openplay_xG.merge(openplay_shots, on=['matchId', 'teamId'])
    openplay_xG['xG_per_shot'] = openplay_xG['expectedGoals'] / openplay_xG['shotCount']
    openplay_xG = openplay_xG.rename(columns={'expectedGoals': 'openplay_xG_liv','xG_per_shot':'xG_per_shot_Liv'})
    setpiece_xG = setpiece_xG.rename(columns={'expectedGoals': 'setpiece_xG_liv'})
    npxG = npxG.rename(columns={'expectedGoals': 'npxG_liv'})


    #--- Opponents
    npxG_Opp = opponents_data[opponents_data['situation']!='Penalty'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()

    openplay_xG_Opp = opponents_data[opponents_data['situation']=='RegularPlay'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()
    setpiece_xG_Opp = opponents_data[opponents_data['situation']=='SetPiece'].groupby(['matchId','teamId'])['expectedGoals'].sum().reset_index()
    openplay_shots_Opp = opponents_data[opponents_data['situation']=='RegularPlay'].groupby(['matchId', 'teamId']).size().reset_index(name='shotCount')
    openplay_xG_Opp = openplay_xG_Opp.merge(openplay_shots_Opp, on=['matchId', 'teamId'])
    openplay_xG_Opp['xG_per_shot'] = openplay_xG_Opp['expectedGoals'] / openplay_xG_Opp['shotCount']

    openplay_xG_Opp = openplay_xG_Opp.rename(columns={'expectedGoals': 'openplay_xG_Opp','xG_per_shot':'OP_xG/shots_Opp'})
    setpiece_xG_Opp = setpiece_xG_Opp.rename(columns={'expectedGoals': 'setpiece_xG_Opp'})
    npxG_Opp = npxG_Opp.rename(columns={'expectedGoals': 'npxG_Opp'})

    # Merge dataframes for Liverpool
    liv_merged = npxG.merge(setpiece_xG, on=['matchId', 'teamId']).merge(openplay_xG, on=['matchId', 'teamId']).merge(xG, on=['matchId', 'teamId'])


    opp_merged_df = npxG_Opp.merge(setpiece_xG_Opp, on=['matchId', 'teamId']).merge(openplay_xG_Opp, on=['matchId', 'teamId'])

    # Add match_date to the dataframes
    liv_merged = liv_merged.merge(data[['matchId', 'match_date']], on=['matchId'])
    opp_merged_df = opp_merged_df.merge(data[['matchId', 'match_date']], on=['matchId'])


    return liv_merged,opp_merged_df
#%%
stats =calculate_match_shots_stats(shots_data,8650)[0]

stats_opp = calculate_match_shots_stats(shots_data,8650)[1]
#%%




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
    opta_data[col] = opta_data[col].astype(bool)


#%%
def calculate_match_stats(data, teamId):
    liv_data = data[data['teamId'] == teamId]
    team_matches = liv_data['matchId'].unique()
    opp_data = data[(data['matchId'].isin(team_matches)) & (data['teamId'] != teamId)]


    final_third = data[data['x'] >= 60]
    defensive_actions = final_third[final_third['event_type'].isin(
        ['BallRecovery', 'BlockedPass', 'ChallengeWon', 'Clearance', 'Foul', 'Interception', 'TackleWon'])]
    defensive_actions_count = defensive_actions.groupby(['matchId', 'teamId'])['eventId'].count().reset_index(
        name='defensive_actions_count')
    opponent_passes = final_third[final_third['event_type'] == 'Pass']
    opponent_passes_count = opponent_passes.groupby(['matchId', 'teamId'])['eventId'].count().reset_index(
        name='opponent_passes_count')
    ppda_data = pd.merge(defensive_actions_count, opponent_passes_count, on=['matchId', 'teamId'])
    ppda_data['PPDA'] = ppda_data['opponent_passes_count'] / ppda_data['defensive_actions_count']


    # Liverpool stats
    liv_xthreat = liv_data.groupby(['matchId', 'team_name', 'teamId'])['xThreat_gen'].sum().reset_index()
    liv_passes = liv_data[liv_data['event_type'] == 'Pass']
    liv_successful_passes = liv_passes[liv_passes['outcomeType'] == 'Successful'].groupby('matchId').count()
    liv_total_passes = liv_passes.groupby('matchId').count()
    liv_pass_success_rate = (liv_successful_passes['id'] / liv_total_passes['id']).reset_index()
    liv_pass_success_rate.columns = ['matchId', 'pass_success_rate']


    defensive_actions = data[data['event_type'].isin(
        ['BallRecovery', 'BlockedPass', 'ChallengeWon', 'Clearance', 'Foul', 'Interception', 'TackleWon'])]
    defensive_line_height = 100 - defensive_actions.groupby(['matchId'])['endY'].mean()
    liv_defensive_line_height = defensive_line_height.reset_index(name='defensive_line_height')

    liv_merged_df = liv_xthreat.merge(liv_pass_success_rate, on='matchId').merge(liv_defensive_line_height, on='matchId')


    # Opponent stats
    opp_xthreat = opp_data.groupby(['matchId', 'team_name', 'teamId'])['xThreat_gen'].sum().reset_index()
    opp_passes = opp_data[opp_data['event_type'] == 'Pass']
    opp_successful_passes = opp_passes[opp_passes['outcomeType'] == 'Successful'].groupby('matchId').count()
    opp_total_passes = opp_passes.groupby('matchId').count()
    opp_pass_success_rate = (opp_successful_passes['id'] / opp_total_passes['id']).reset_index()
    opp_pass_success_rate.columns = ['matchId', 'pass_success_rate']
    opp_merged_df = opp_xthreat.merge(opp_pass_success_rate, on='matchId')

    # Add match_date to the dataframes
    liv_merged_df = liv_merged_df.merge(data[['matchId', 'match_date']], on=['matchId'])
    opp_merged_df = opp_merged_df.merge(data[['matchId', 'match_date']], on=['matchId'])


    return liv_merged_df,opp_merged_df,ppda_data
#%%
touches = opta_data[opta_data['isTouch'] == True]
possession_metric = touches.groupby(['matchId', 'teamId']).size() / touches.groupby('matchId').size()
possession_metric = possession_metric.reset_index(name='possession_metric')

#%%
match_stats = calculate_match_stats(data=opta_data,teamId=26)[0]
match_stats_opp =calculate_match_stats(data=opta_data,teamId=26)[1]
ppda = calculate_match_stats(data=opta_data,teamId=26)[2]


#%%
def get_match_name(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
    data = json.loads(response.content)
    general = data['general']
    Hteam = general['homeTeam']
    Ateam = general['awayTeam']
    Hteam = Hteam['name']
    Ateam = Ateam['name']
    return Hteam + " " + "vs" + " " + Ateam


match_name = get_match_name(Fotmob_matchID)


def get_match_score(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}')
    data = json.loads(response.content)
    match_score = data['header']['status']['scoreStr']
    return match_score


match_score = get_match_score(Fotmob_matchID)
#%%
fig, axs = plt.subplots(nrows=16, ncols=1, figsize=(3, 4), dpi=900)
fig.set_facecolor("#201D1D")
fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.8, wspace=0.2, hspace=0.5)



team_logo_path = f'Post_Match_Dashboard/Data/team_logo/{8650}.png'
club_icon = Image.open(team_logo_path).convert('RGBA')


logo_ax = fig.add_axes([-0.04, .82, 0.08, 0.08], frameon=False)

logo_ax.imshow(club_icon, aspect='equal')
logo_ax.set_xticks([])
logo_ax.set_yticks([])

fig_text(
    0.4,
    0.88,
    match_score,
    fontsize=5,
    color="#FCE6E6",
    ha="center",
    va="center",
    # transform=ax.transAxes
)

fig_text(
    0.4,
    0.86,
    match_name,
    fontsize=4,
    color="#FCE6E6",
    ha="center",
    va="center",
    # transform=ax.transAxes
)

fig_text(
    0.4,
    0.84,
    f'Compared to Liverpool\'s {comp_name} Average since the start of the 2023/24 season',
    fontsize=3,
    color="#FCE6E6",
    ha="center",
    va="center",
    # transform=ax.transAxes
)




axs = axs.flatten()
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_facecolor("#212529")
    # ax.set_xlim(x_lower_bound, x_upper_bound)

# Create the scatter plot
df_scatter = pd.DataFrame()
for index, match in enumerate(stats['matchId']):
    df_aux = stats[stats['matchId'] == match]
    # df_aux = df_aux.assign(index)
    df_aux = df_aux.assign(index=index)
    df_scatter = pd.concat([df_scatter, df_aux])
    df_scatter.reset_index(drop=True, inplace=True)

axs[0].set_ylabel('npxG', fontsize=3, color='white', rotation='horizontal', labelpad=16)
axs[1].set_ylabel('Opp npxG', fontsize=3, color='white', rotation='horizontal', labelpad=16)

axs[2].set_ylabel('SetPiece xG', fontsize=3, color='white', rotation=0, labelpad=16)

axs[3].set_ylabel('Opp SetPiece xG', fontsize=3, color='white', rotation=0, labelpad=16)

axs[4].set_ylabel('Open-play xG', fontsize=3, color='white', rotation='horizontal', labelpad=16)
axs[5].set_ylabel('Opp Open-play xG', fontsize=3, color='white', rotation='horizontal', labelpad=16)

axs[6].set_ylabel('Open-play xG /Shot', fontsize=3, color='white', rotation=0, labelpad=16)

axs[7].set_ylabel('Opp Open-play xG /Shot', fontsize=3, color='white', rotation=0, labelpad=18)

axs[8].set_ylabel('xThreat', fontsize=3, color='white', rotation='horizontal', labelpad=16)
axs[9].set_ylabel('Opp xThreat', fontsize=3, color='white', rotation=0, labelpad=16)
axs[10].set_ylabel('PPDA', fontsize=3, color='white', rotation='horizontal', labelpad=16)
axs[11].set_ylabel('Defensive line height ', fontsize=3, color='white', rotation=0, labelpad=16)
# axs[12].set_ylabel('Field Tilt', fontsize=3, color='white', rotation=0, labelpad=16)
axs[13].set_ylabel('Possession', fontsize=3, color='white', rotation=0, labelpad=16)
axs[14].set_ylabel('Pass Completion %', fontsize=3, color='white', rotation='horizontal', labelpad=20)
axs[15].set_ylabel('Opp Pass Completion %', fontsize=3, color='white', rotation=0, labelpad=20)

# axs[16].set_ylabel('Counter Attacks xG', fontsize=3, color='white', rotation='horizontal', labelpad=20)
# axs[17].set_ylabel('Opp Counter Attacks xG ', fontsize=3, color='white', rotation=0, labelpad=20)

axs[12].set_ylabel('xGOT', fontsize=3, color='white', rotation=0, labelpad=14)

sns.scatterplot(data=stats_opp, x='npxG_Opp', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[1])
sns.scatterplot(data=stats_opp[stats_opp['matchId'] == Fotmob_matchID], x='npxG_Opp', y=index, c='#660708', edgecolor='k', s=20,
                marker='o', alpha=.88, ax=axs[1])


sns.scatterplot(data=stats, x='openplay_xG_liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[4])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='openplay_xG_liv', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[4])


sns.scatterplot(data=stats_opp, x='openplay_xG_Opp', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[5])
sns.scatterplot(data=stats_opp[stats_opp['matchId'] == Fotmob_matchID], x='openplay_xG_Opp', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[5])

sns.scatterplot(data=stats, x='setpiece_xG_liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[2])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='setpiece_xG_liv', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[2])


sns.scatterplot(data=stats_opp, x='setpiece_xG_Opp', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[3])
sns.scatterplot(data=stats_opp[stats_opp['matchId'] == Fotmob_matchID], x='setpiece_xG_Opp', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[3])



sns.scatterplot(data=stats, x='npxG_liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[0])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='npxG_liv', y=index, c='#660708', edgecolor='k', s=20,
                marker='o', alpha=.88, ax=axs[0])


sns.scatterplot(data=stats, x='xG_per_shot_Liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[6])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='xG_per_shot_Liv', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[6])

sns.scatterplot(data=stats_opp, x='OP_xG/shots_Opp', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[7])
sns.scatterplot(data=stats_opp[stats_opp['matchId'] == Fotmob_matchID], x='OP_xG/shots_Opp', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[7])


sns.scatterplot(data=match_stats_opp[match_stats_opp['teamId']!=26], x='xThreat_gen', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[9])
sns.scatterplot(data=match_stats[match_stats['teamId']==26], x='xThreat_gen', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[8])

sns.scatterplot(data=match_stats_opp[(match_stats_opp['teamId']!=26) & (match_stats_opp['matchId'] == opta_matchID)], x='xThreat_gen', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[9])
sns.scatterplot(data=match_stats[(match_stats['teamId']==26) & (match_stats['matchId'] == opta_matchID)], x='xThreat_gen', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[8])


sns.scatterplot(data=ppda[ppda['teamId']==26], x='PPDA', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.88,
                ax=axs[10])

sns.scatterplot(data=ppda[(ppda['teamId']==26) & (ppda['matchId'] == opta_matchID)], x='PPDA', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[10])



sns.scatterplot(data=match_stats[(match_stats['teamId']==26) & (match_stats['matchId'] == opta_matchID)], x='defensive_line_height', y=index, c='#660708', edgecolor='k', s=20, marker='o',zorder=4, alpha=.88,
                ax=axs[11])

sns.scatterplot(data=match_stats[match_stats['teamId']==26], x='defensive_line_height', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[11])

sns.scatterplot(data=possession_metric[(possession_metric['teamId']==26) & (possession_metric['matchId'] == opta_matchID)], x='possession_metric', y=index, c='#660708', edgecolor='k', s=20, marker='o',zorder=4, alpha=.88,
                ax=axs[13])

sns.scatterplot(data=possession_metric[possession_metric['teamId']==26], x='possession_metric', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.88,
                ax=axs[13])

sns.scatterplot(data=match_stats[match_stats['teamId']==26], x='pass_success_rate', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[14])

sns.scatterplot(data=match_stats[(match_stats['teamId']==26) & (match_stats['matchId'] == opta_matchID)], x='pass_success_rate', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[14])


sns.scatterplot(data=match_stats_opp[match_stats_opp['teamId']!=26], x='pass_success_rate', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[15])
sns.scatterplot(data=match_stats_opp[(match_stats_opp['teamId']!=26) & (match_stats_opp['matchId'] == opta_matchID)], x='pass_success_rate', y=index, c='#660708', edgecolor='k', s=20, marker='o', alpha=.88, ax=axs[15])


sns.scatterplot(data=stats, x='xGOT_liv', y=index, c='#43B8AA', edgecolor='#43B8AA', s=20, marker='o', alpha=.2,
                ax=axs[12])
sns.scatterplot(data=stats[stats['matchId'] == Fotmob_matchID], x='xGOT_liv', y=index, c='#660708', edgecolor='k',
                s=20, marker='o', alpha=.88, ax=axs[12])

# fig.savefig(f"app/Post_Match_Dashboard/figures/match_avgDashboard{today}.png", dpi=900, bbox_inches="tight")



# Create a BytesIO object to store the figure
figure_buffer = BytesIO()

# Save the figure to the BytesIO object
plt.savefig(
    figure_buffer,
    format="png",  # Use the appropriate format for your figure
    dpi=700,
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
blob_path = f"figures/{today}/match_avgDashboard{today}.png"

# Create a new Blob and upload the figure
blob = bucket.blob(blob_path)
blob.upload_from_file(figure_buffer, content_type="image/png")

# Close the BytesIO buffer
figure_buffer.close()
