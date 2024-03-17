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
from sqlalchemy import create_engine,exc
from datetime import datetime
import pytz

user = os.environ['PGUSER']
passwd = os.environ['PGPASSWORD']
host = os.environ['PGHOST']
port = os.environ['PGPORT']
db = os.environ['PGDATABASE']
db_url = f'postgresql://{user}:{passwd}@{host}:{port}/{db}'
engine = create_engine(db_url)

parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_mount_path = '/Post_Match_Dashboard/pipeline/'
secret_relative_path = 'scraper-key'
key_file_path = os.path.join(parent_directory, parent_mount_path, secret_relative_path)
if os.path.exists(key_file_path):
    with open(key_file_path, 'r') as key_file:
        key_data = key_file.read()
    key_json = json.loads(key_data)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path
else:
    print("Error: Scraper key file not found at", key_file_path)

storage_client = storage.Client()

eastern_timezone = pytz.timezone('America/New_York')
current_time_in_boston = datetime.now(eastern_timezone)
today_date = current_time_in_boston.strftime("%d-%m-%y")
match_date = today_date
match_id = get_match_date_and_id(8650)[1]

bucket_name = "soccerdb-data"
blob_name = f"{match_date}.json"
data_blob = storage_client.bucket(bucket_name).blob(blob_name)
data_as_text = data_blob.download_as_text()
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
csv_blob = storage_client.bucket(bucket_name).blob(csv_filename)
csv_blob.upload_from_filename(csv_filename)

import pandas as pd

def match_exists_in_db(table_name,date,engine):
    query = f"SELECT * FROM {table_name} WHERE match_date = '{date}' "
    try:
        data = pd.read_sql(query, engine)
        return len(data) > 0
    except exc.ProgrammingError as e:
        return False

date = current_time_in_boston.strftime("%Y-%m-%d")

opta_exists = match_exists_in_db('opta_event_data', date, engine)
fotmob_exists = match_exists_in_db('fotmob_shots_data', date, engine)

if not opta_exists or not fotmob_exists:
    events['qualifiers'] = events['qualifiers'].apply(json.dumps)
    events['satisfiedEventsTypes'] = events['satisfiedEventsTypes'].apply(json.dumps)
    if not opta_exists:
        events.to_sql('opta_event_data', engine, if_exists='append', index=False, dtype={"qualifiers": sqlalchemy.types.JSON, "satisfiedEventsTypes": sqlalchemy.types.JSON})
    if not fotmob_exists:
        data_shots.to_sql('fotmob_shots_data', engine, if_exists='append', index=False)
