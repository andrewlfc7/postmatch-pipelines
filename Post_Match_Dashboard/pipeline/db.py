from ast import literal_eval
from time import sleep

import pandas as pd
import subprocess
import datetime
import yaml

import sqlalchemy

from fotmob import get_shots_data
from schedule import get_match_date_and_id
import json
from processing import *

import os
from google.cloud import storage



import os
import json

# Get the parent directory of the script's directory
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Specify the parent directory path where the secret is mounted
parent_mount_path = '/app/Post_Match_Dashboard/pipeline/'

# Specify the path relative to the parent mount path where the secret file is located
secret_relative_path = 'scraper-key'

# Construct the full path to the secret file
key_file_path = os.path.join(parent_directory, parent_mount_path, secret_relative_path)

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


# Initialize Google Cloud Storage client
storage_client = storage.Client()


from datetime import datetime
import pytz

# Set the timezone to Eastern Time (Boston)
eastern_timezone = pytz.timezone('America/New_York')

# Get the current time in the Eastern Time zone
current_time_in_boston = datetime.now(eastern_timezone)

# Get today's date
today_date = current_time_in_boston.strftime("%d-%m-%y")


match_date = today_date

match_id = get_match_date_and_id(8650)[1]



# Specify the bucket and blob name
bucket_name = "soccerdb-data"
blob_name = f"{match_date}.json"  # Change the extension to json if that's the format you're working with

# Create a blob with the specified name in the specified bucket
data_blob = storage_client.bucket(bucket_name).blob(blob_name)

# Download data as text
data_as_text = data_blob.download_as_text()

# Now, you can process the data_as_text as needed
# For example, you can load it as JSON
data = json.loads(data_as_text)

data = data_processing(data)


data.to_csv(f"data_processed{match_date}.csv")

data_processed_v2 = pd.read_csv(f"data_processed{match_date}.csv")





from ast import literal_eval
data_processed_v2['qualifiers'] = [literal_eval(x) for x in data_processed_v2['qualifiers']]
data_processed_v2['satisfiedEventsTypes'] = [literal_eval(x) for x in data_processed_v2['satisfiedEventsTypes']]



events = custom_events(data_processed_v2)


data_shots = get_shots_data(match_id)


competition = data_shots['competition'].iloc[0]


events['competition'] = competition


csv_filename = f"processed_data{match_date}.csv"
events.to_csv(csv_filename, index=False)

# Upload the CSV file to GCS
csv_blob = storage_client.bucket(bucket_name).blob(csv_filename)
csv_blob.upload_from_filename(csv_filename)




from sqlalchemy import create_engine,exc

import os


user = os.environ['PGUSER']
passwd = os.environ['PGPASSWORD']
host = os.environ['PGHOST']
port = os.environ['PGPORT']
db = os.environ['PGDATABASE']
db_url = f'postgresql://{user}:{passwd}@{host}:{port}/{db}'

engine = create_engine(db_url)





import pandas as pd

def match_exists_in_db(table_name,date,engine):

    # Write your SQL query to retrieve data
    query = f"SELECT * FROM {table_name} WHERE match_date = '{date}' "

    try:
        # Execute the query and fetch the data into a DataFrame
        data = pd.read_sql(query, engine)
        return len(data) > 0
    except exc.ProgrammingError as e:
        # Catch UndefinedTable error and return False
        return False



date = current_time_in_boston.strftime("%Y-%m-%d")



# Check if the match data already exists in the database tables
opta_exists = match_exists_in_db('opta_event_data', date, engine)
fotmob_exists = match_exists_in_db('fotmob_shots_data', date, engine)

if not opta_exists or not fotmob_exists:
    # Convert specific columns to JSON strings
    events['qualifiers'] = events['qualifiers'].apply(json.dumps)
    events['satisfiedEventsTypes'] = events['satisfiedEventsTypes'].apply(json.dumps)

    # Insert data into 'opta_event_data' table
    if not opta_exists:
        events.to_sql('opta_event_data', engine, if_exists='append', index=False, dtype={"qualifiers": sqlalchemy.types.JSON, "satisfiedEventsTypes": sqlalchemy.types.JSON})

    # Insert data into 'fotmob_shots_data' table
    if not fotmob_exists:
        data_shots.to_sql('fotmob_shots_data', engine, if_exists='append', index=False)


# events['qualifiers'] = events['qualifiers'].apply(json.dumps)
# events['satisfiedEventsTypes'] = events['satisfiedEventsTypes'].apply(json.dumps)
#
#
# events.to_sql('opta_event_data', engine, if_exists='append', index=False, dtype={"qualifiers": sqlalchemy.types.JSON, "satisfiedEventsTypes": sqlalchemy.types.JSON})
#
# data_shots.to_sql('fotmob_shots_data', engine, if_exists='append', index=False)
#
#
#


