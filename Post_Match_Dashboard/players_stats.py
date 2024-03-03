
import datetime
import json
import os
from google.cloud import storage
from io import BytesIO
from unidecode import unidecode
from fuzzywuzzy import fuzz

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import zscore
import re
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
from utils import check_players_in_match
# Set the time zone to Eastern Time
eastern = pytz.timezone('US/Eastern')

# Get the current date in Eastern Time
# today = datetime.datetime.now(eastern).date()
# today = today.strftime('%Y-%m-%d')


today = '2024-03-02'
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




match_date = today

Fotmob_matchID = shots_data[shots_data['match_date'] == match_date]['match_id'].iloc[0]

opta_matchID = opta_data[opta_data['match_date'] == match_date]['match_id'].iloc[0]


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



class PlayerMatchStatsCalculator:
    def __init__(self, event_data, shots_df):
        self.event_data = event_data[event_data['is_open_play'] == True].copy()
        self.shots_df = shots_df.copy()
        self.opta_stats = {}
        self.opta_passes_df = None
        self.opta_events = None
        self.player_name_mapping = {}
        self.apply_name_matching()

    def match_player_name(self, name):
        return self.player_name_mapping.get(name, name)

    def normalize_name(self, name):
        # Remove accents
        name = unidecode(name)
        # Remove leading/trailing whitespaces
        name = name.strip()
        return name

    def match_names(self, data_names, shots_names):
        matched_names = set()
        for data_name in data_names:
            for shots_name in shots_names:
                if fuzz.partial_ratio(data_name, shots_name) > 60 or (data_name.split()[0] == shots_name.split()[0]):
                    matched_names.add((data_name, shots_name))
        return matched_names

    def apply_name_matching(self):
        # Extract unique player names from event_data and shots_df
        names_with_accents_event = set(self.event_data['playerName'])
        names_without_accents_shots = set(self.shots_df['playerName'])

        # Normalize names
        normalized_names_event = {self.normalize_name(name) for name in names_with_accents_event}
        normalized_names_shots = {self.normalize_name(name) for name in names_without_accents_shots}

        # Match names
        matched_names = self.match_names(normalized_names_event, normalized_names_shots)

        # Build name mapping dictionary
        for name_pair in matched_names:
            self.player_name_mapping[name_pair[0]] = name_pair[1]

        # Apply name mapping to event_data and shots_df
        self.event_data['playerName'] = self.event_data['playerName'].apply(self.match_player_name)
        self.shots_df['playerName'] = self.shots_df['playerName'].apply(self.match_player_name)

    def calculate_event_stats(self):
        # event_types = [
        #     'BallRecovery',
        #     'Aerial',
        #     'Tackle'
        # ]
        # combined_stats_df = pd.DataFrame(columns=['opta_matchID', 'match_date', 'playerName'])
        # for event_type in event_types:
        #     filtered_data = self.event_data[(self.event_data['event_type'] == event_type) & (self.event_data['outcomeType'] == 'Successful')]
        #     opta_counts = filtered_data.groupby(['opta_matchID', 'match_date', 'playerName']).size().reset_index()
        #     opta_counts.columns = ['opta_matchID', 'match_date', 'playerName', event_type]
        #     combined_stats_df = pd.merge(combined_stats_df, opta_counts, how='outer', on=['opta_matchID', 'match_date', 'playerName'])
        # combined_stats_df['opta_matchID'] = combined_stats_df['opta_matchID'].astype(int)
        # combined_stats_df.fillna(0, inplace=True)
        # combined_stats_df.reset_index(drop=True, inplace=True)
        #
        event_types = [
            'BallRecovery',
            'Aerial',
            'Tackle'
        ]

        combined_stats_df = pd.DataFrame(columns=['opta_matchID', 'match_date', 'playerName'])

        for event_type in event_types:
            if event_type == 'BallRecovery':
                new_name = 'Ball Recoveries'
            elif event_type == 'Aerial':
                new_name = 'Aerials Won'
            elif event_type == 'Tackle':
                new_name = 'Tackles Won'

            filtered_data = self.event_data[(self.event_data['event_type'] == event_type) & (self.event_data['outcomeType'] == 'Successful')]
            opta_counts = filtered_data.groupby(['opta_matchID', 'match_date', 'playerName']).size().reset_index()
            opta_counts.columns = ['opta_matchID', 'match_date', 'playerName', new_name]
            combined_stats_df = pd.merge(combined_stats_df, opta_counts, how='outer', on=['opta_matchID', 'match_date', 'playerName'])

        combined_stats_df['opta_matchID'] = combined_stats_df['opta_matchID'].astype(int)
        combined_stats_df.fillna(0, inplace=True)
        combined_stats_df.reset_index(drop=True, inplace=True)


        self.opta_events = combined_stats_df
        # Calculate defensive actions
        defensive_actions = (
                (self.event_data['event_type'] == 'Interception') & (self.event_data['outcomeType'] == 'Successful') |
                (self.event_data['event_type'] == 'Clearance') & (self.event_data['outcomeType'] == 'Successful') |
                (self.event_data['event_type'] == 'Challenge') & (self.event_data['outcomeType'] == 'Successful') |
                (self.event_data['event_type'] == 'SavedShot') & (self.event_data['outcomeType'] == 'Successful') |
                (self.event_data['event_type'] == 'Save') & (self.event_data['outcomeType'] == 'Successful') |
                (self.event_data['event_type'] == 'BlockedPass') & (self.event_data['outcomeType'] == 'Successful')
        )
        defensive_actions_df = self.event_data[defensive_actions].groupby(['opta_matchID', 'match_date', 'playerName']).size().reset_index(name='Defensive Actions')

        # Merge defensive actions with combined_stats_df
        self.opta_events = pd.merge(combined_stats_df, defensive_actions_df, how='left', on=['opta_matchID', 'match_date', 'playerName'])
        self.opta_events['Defensive Actions'].fillna(0, inplace=True)

    def get_pass_stats(self):
        def get_passes_df(events_dict):
            df = pd.DataFrame(events_dict)

            # create receiver column based on the next event
            # this will be correct only for successful passes
            df["pass_recipient"] = df["playerName"].shift(-1)
            # filter only passes
            passes_ids = df.index[df['event_type'] == 'Pass']
            df_passes = df.loc[passes_ids, ["id", "match_date", "minute", "x", "y", "endX", "endY", "opta_matchID", "teamId", "playerId",
                                            "playerName", "event_type", "outcomeType", "pass_recipient", 'isTouch',
                                            'xThreat_gen', 'is_progressive', 'FinalThirdPasses']]

            return df_passes

        passes_df = get_passes_df(self.event_data)
        passes_df = passes_df[passes_df['outcomeType'] == 'Successful']

        pass_between = passes_df.groupby(['match_date', 'opta_matchID', 'playerName', 'pass_recipient', 'is_progressive',
                                          'FinalThirdPasses', 'xThreat_gen']).agg(
            total_xt=('xThreat_gen', 'sum'),
            pass_count=('id', 'count')).reset_index()

        result = pass_between.groupby(['opta_matchID', 'match_date', 'pass_recipient']).agg(
            total_xt=('total_xt', 'sum'),
        ).reset_index()

        total_passes_received_final_third = passes_df[passes_df['FinalThirdPasses'] == True].groupby(
            ['opta_matchID', 'match_date', 'pass_recipient']).size()
        total_passes_received_final_third = total_passes_received_final_third.rename('Final Third Passes Receive')

        total_passes_received_progressive = passes_df[passes_df['is_progressive'] == True].groupby(
            ['opta_matchID', 'match_date', 'pass_recipient']).size()
        total_passes_received_progressive = total_passes_received_progressive.rename('Progressive Passes Receive')

        total_xThreat_received = passes_df.groupby(['opta_matchID', 'match_date', 'pass_recipient'])['xThreat_gen'].sum()
        total_xThreat_received = total_xThreat_received.rename('xThreat received')

        result = result.join(total_passes_received_final_third, on=['opta_matchID', 'match_date', 'pass_recipient'])
        result = result.join(total_passes_received_progressive, on=['opta_matchID', 'match_date', 'pass_recipient'])
        result = result.join(total_xThreat_received, on=['opta_matchID', 'match_date', 'pass_recipient'])

        result = result[['opta_matchID', 'match_date', 'pass_recipient',
                         'Final Third Passes Receive',
                         'Progressive Passes Receive', 'xThreat received']]

        result.columns = ['opta_matchID', 'match_date', 'playerName',
                          'Final Third Passes Receive',
                          'Progressive Passes Receive', 'xThreat received']

        self.opta_passes_df = result

    def calculate_additional_stats(self):

        self.opta_stats['Progressive Passes'] = self.event_data[self.event_data['is_progressive'] == True].groupby(['opta_matchID', 'match_date', 'playerName']).size()
        self.opta_stats['Final Third Passes'] = self.event_data[self.event_data['FinalThirdPasses'] == True].groupby(['opta_matchID', 'match_date', 'playerName']).size()
        self.opta_stats['Key Passes'] = self.event_data[self.event_data['key_pass'] == True].groupby(['opta_matchID', 'match_date', 'playerName']).size()

        pass_successful = self.event_data[self.event_data['event_type'] == 'Pass'].groupby(['opta_matchID', 'match_date', 'playerName'])['outcomeType'].apply(lambda x: (x == 'Successful').sum())
        pass_total = self.event_data[self.event_data['event_type'] == 'Pass'].groupby(['opta_matchID', 'match_date', 'playerName']).size()
        self.opta_stats['Passes Completed %'] = pass_successful / pass_total * 100
        self.opta_stats['Passes'] = pass_successful
        self.opta_stats['Carries'] = self.event_data[(self.event_data['event_type'] == 'Carry') & (self.event_data['progressive_carry'] == True)].groupby(['opta_matchID', 'match_date', 'playerName']).size()

        self.opta_stats['xThreatGen'] = self.event_data.groupby(['opta_matchID', 'match_date', 'playerName'])['xThreat_gen'].sum()

    def merge_stats(self):
        merged_data = pd.merge(self.opta_events, self.opta_passes_df, how='outer', on=['playerName', 'opta_matchID', 'match_date'])
        merged_data = pd.merge(merged_data, pd.DataFrame(self.opta_stats), how='outer', on=['playerName', 'opta_matchID', 'match_date'])
        merged_data.fillna(0, inplace=True)
        merged_data.reset_index(drop=True, inplace=True)
        merged_data['playerName'] = merged_data['playerName'].apply(unidecode)  # Apply unidecode to playerName column
        return merged_data

    def calculate(self):
        self.calculate_event_stats()
        self.get_pass_stats()
        self.calculate_additional_stats()
        # Incorporating Fotmob stats
        fotmob_stats = {}
        fotmob_stats['Shots'] = self.shots_df.groupby(['Fotmob_matchID', 'match_date', 'playerName']).size()
        # fotmob_stats['Shots on Target %'] = self.shots_df[(self.shots_df['isOnTarget'] == True) & (self.shots_df['eventType'] == 'Goal')].groupby(['Fotmob_matchID', 'match_date', 'playerName']).size()
        # Calculate shots on target
        shots_on_target = self.shots_df[(self.shots_df['isOnTarget'] == True) & (self.shots_df['eventType'] == 'Goal')].groupby(['Fotmob_matchID', 'match_date', 'playerName']).size()
        # Calculate percentage of shots on target
        fotmob_stats['Shots on Target %'] = (shots_on_target / fotmob_stats['Shots']) * 100
        fotmob_stats['Goals'] = self.shots_df[self.shots_df['eventType'] == 'Goal'].groupby(['Fotmob_matchID', 'match_date', 'playerName']).size()
        fotmob_stats['xG'] = self.shots_df.groupby(['Fotmob_matchID', 'match_date', 'playerName'])['expectedGoals'].sum().round(2)
        self.shots_df['playerName'] = self.shots_df['playerName'].apply(unidecode)
        # Merging Fotmob stats with Opta stats
        merged_data = self.merge_stats()
        merged_data = pd.merge(merged_data, pd.DataFrame(fotmob_stats), how='outer', on=['playerName', 'match_date'])
        merged_data.fillna(0, inplace=True)
        return merged_data



shots_data = shots_data.rename(columns={'match_id':'Fotmob_matchID'})
opta_data = opta_data.rename(columns={'match_id':'opta_matchID'})


calculator = PlayerMatchStatsCalculator(opta_data, shots_data)
merged_stats = calculator.calculate()


class PlayerStatsVisualizer:
    def __init__(self, stats, playername, player_position):
        self.stats = stats
        self.playername = playername
        self.player_position = player_position
        self.position_columns = {
            "Attackers": [
                "Shots",
                "Shots on Target %",
                "xG",
                # "xA",
                "Goals",
                "Progressive Passes Receive",
                "Final Third Passes Receive",
                "Passes",
                "Passes Completed %",
                "Carries",
                "Key Passes",
                "Tackles Won",
                "Ball Recoveries",
                "Defensive Actions",
                "Aerials Won",
                "match_date",
                "opta_matchID"
            ],
            "Midfielder":  [
                "Passes Completed %",
                "Passes",
                "Progressive Passes",
                "Final Third Passes",
                "Key Passes",
                "Shots",
                # "xA",
                "xG",
                "Tackles Won",
                "Ball Recoveries",
                "Defensive Actions",
                "Aerials Won",
                "match_date",
                "opta_matchID"
            ],
            "Defenders": [
                "Tackles Won",
                "Ball Recoveries",
                "Defensive Actions",
                "Aerials Won",
                "Passes Completed %",
                "Passes",
                "Progressive Passes",
                "Final Third Passes",
                "Key Passes",
                "Shots",
                # "xA",
                "xG",
                "match_date",
                "opta_matchID"
            ]
        }

    def plot_stats(self):
        columns = self.position_columns.get(self.player_position, [])
        if not columns:
            print("Position columns not found for the given player position.")
            return

        filtered_stats = self.stats[(self.stats['playerName'] == self.playername)][columns]

        if filtered_stats.empty:
            print("No data found for the given player.")
            return

        # Z-score normalization
        normalized_data = filtered_stats[columns[:-3]].apply(zscore)

        # Handling NaN and Inf values
        normalized_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        normalized_data.dropna(inplace=True)

        fig, axes = plt.subplots(nrows=len(columns)-2, ncols=1, figsize=(6, 8), dpi=300)
        fig.set_facecolor("#201D1D")
        fig.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.88, wspace=0.2, hspace=0.5)

        df_scatter = pd.DataFrame()
        for index, match in enumerate(filtered_stats['opta_matchID']):
            df_aux = filtered_stats[filtered_stats['opta_matchID'] == match]
            df_aux = df_aux.assign(index=index)
            df_scatter = pd.concat([df_scatter, df_aux])
            df_scatter.reset_index(drop=True, inplace=True)

        excluded_columns = ['opta_matchID', 'match_date', 'playerName']
        for i, column in enumerate(filtered_stats.columns):
            if column not in excluded_columns:
                ax = axes[i]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.set_facecolor("#212529")

                column_label = (lambda x: re.sub(r'([a-z])([A-Z])', r'\1 \2', x))(column)
                ax.set_ylabel(column_label, fontsize=5, color='white', rotation=0,fontweight='bold', labelpad=38)

                sns.scatterplot(data=df_scatter, x=column, y=index, c='#43B8AA', edgecolor='#43B8AA', s=40, marker='o', alpha=.16, ax=ax)


                # Highlight current match using match_date
                current_match_date = df_scatter['match_date'].iloc[-1]

                # Scatter plot for current match
                current_match_data = df_scatter[df_scatter['match_date'] == current_match_date]
                sns.scatterplot(data=current_match_data, x=column, y=index, c='#660708', edgecolor='k', s=40, marker='o', alpha=.8, ax=ax)

                # Adding values as text on scatter points (highlighted current match)
                for x, y, value in zip(current_match_data[column], current_match_data['index'], current_match_data[column]):
                    ax.annotate(
                        f'{round(value, 1)}',
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, -8),
                        ha='center',
                        fontsize=6,
                        color='white',
                        fontweight='bold',

                    )

                # # Set x-axis limits using z-score normalized data for each subplot
                # if not np.isnan(np.min(normalized_data[column])) and not np.isnan(np.max(normalized_data[column])):
                #     ax.set_xlim(np.min(normalized_data[column]), np.max(normalized_data[column]))
                # else:
                #     ax.set_xlim(-3, 3)  # Default range if NaN or Inf values are found



        fig_text(
            0.4,
            0.94,
            match_name,
            fontsize=6,
            color="#FCE6E6",
            ha="center",
            va="center",
            fontweight='bold'
            # transform=ax.transAxes
        )
        fig_text(
            0.4,
            0.92,
            match_score,
            fontsize=6,
            color="#FCE6E6",
            ha="center",
            va="center",
            fontweight='bold'
            # transform=ax.transAxes
        )
        fig.text(
            0.42,
            0.90,
            f'Compared to {self.playername}\'s {comp_name} Average since the start of the 2023/24 season',
            fontsize=6,
            color="#FCE6E6",
            ha="center",
            va="center",
            fontweight='bold'
            # transform=axes[-1].transAxes
        )
        return plt



fw = [
    "Cody Gakpo",
    "Diogo Jota",
    "Mohamed Salah",
    "Darwin Nunez",
    "Luis Diaz"
]

mf = [
    "Alexis Mac Allister",
    "Curtis Jones",
    "Dominik Szoboszlai",
    "Ryan Gravenberch",
    "Wataru Endo",
    "Harvey Elliott"
]





df = [
    "Ibrahima Konate",
    "Virgil van Dijk",
    "Konstantinos Tsimikas",
    "Joel Matip",
    "Jarell Quansah",
    "Andrew Robertson",
    "Conor Bradley"
]



df_in_match = check_players_in_match(df, opta_data[opta_data['match_date']==today])
mf_in_match = check_players_in_match(mf, opta_data[opta_data['match_date']==today])
fw_in_match = check_players_in_match(fw, opta_data[opta_data['match_date']==today])




for player in fw_in_match:
    try:
        # Assuming PlayerStatsVisualizer class is defined elsewhere
        player_stats_visualizer = PlayerStatsVisualizer(merged_stats, player, "Attackers")
        player_stats_visualizer.plot_stats()

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
        blob_path = f"figures/{today}/{player.replace(' ', '_')}_avgDashboard{today}.png"

        # Create a new Blob and upload the figure
        blob = bucket.blob(blob_path)
        blob.upload_from_file(figure_buffer, content_type="image/png")

        # Close the figure buffer
        figure_buffer.close()

    except Exception as e:
        print(f"Error occurred for player {player}: {str(e)}")


for player in mf_in_match:
    try:
        # Assuming PlayerStatsVisualizer class is defined elsewhere
        player_stats_visualizer = PlayerStatsVisualizer(merged_stats, player, "Midfielder")
        player_stats_visualizer.plot_stats()

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

        # Get today's date (assuming 'today' is defined elsewhere)

        # Specify the blob path within the bucket
        blob_path = f"figures/{today}/{player.replace(' ', '_')}_avgDashboard{today}.png"

        # Create a new Blob and upload the figure
        blob = bucket.blob(blob_path)
        blob.upload_from_file(figure_buffer, content_type="image/png")

        # Close the figure buffer
        figure_buffer.close()

    except Exception as e:
        print(f"Error occurred for player {player}: {str(e)}")

for player in df_in_match:
    try:
        # Assuming PlayerStatsVisualizer class is defined elsewhere
        player_stats_visualizer = PlayerStatsVisualizer(merged_stats, player, "Defenders")
        player_stats_visualizer.plot_stats()

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
        blob_path = f"figures/{today}/{player.replace(' ', '_')}_avgDashboard{today}.png"

        # Create a new Blob and upload the figure
        blob = bucket.blob(blob_path)
        blob.upload_from_file(figure_buffer, content_type="image/png")

        # Close the figure buffer
        figure_buffer.close()

    except Exception as e:
        print(f"Error occurred for player {player}: {str(e)}")