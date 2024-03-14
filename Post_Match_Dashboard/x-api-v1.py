import os
import tweepy
import datetime
import pytz
from google.cloud import storage
from utils import get_match_name
import json
import pandas as pd
from sqlalchemy import create_engine

eastern = pytz.timezone('US/Eastern')
today = datetime.datetime.now(eastern).date()
today = today.strftime('%Y-%m-%d')

key_file_path = '/Post_Match_Dashboard/postmatch-key'

if os.path.exists(key_file_path):
    with open(key_file_path, 'r') as key_file:
        key_data = key_file.read()
    key_json = json.loads(key_data)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = key_file_path
else:
    print("Error: postmatch key file not found at", key_file_path)


user = os.environ['PGUSER']
passwd = os.environ['PGPASSWORD']
host = os.environ['PGHOST']
port = os.environ['PGPORT']
db = os.environ['PGDATABASE']
db_url = f'postgresql://{user}:{passwd}@{host}:{port}/{db}'

engine = create_engine(db_url)

conn = engine.connect()

shots_query =f"""
SELECT * FROM fotmob_shots_data WHERE match_date = '{today}' AND ("teamId" = 8650 OR "match_id" IN (SELECT "match_id" FROM fotmob_shots_data WHERE "teamId" = 8650))

"""


shots_data = pd.read_sql(shots_query, conn)

Fotmob_matchID = shots_data['match_id'].iloc[0]



def verify_twitter_credentials():
    """Verify twitter authentication"""

    consumer_key = os.environ['API_KEY']
    consumer_secret = os.environ['API_SECRET']
    access_token = os.environ['ACCESS_TOKEN']
    access_secret_token = os.environ['ACCESS_TOKEN_SECRET']

    api = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_secret_token
    )

    return api

def tweet_images(api: tweepy.Client, images, tweet=''):
    """Upload image to Twitter with a tweet"""

    consumer_key = os.environ['API_KEY']
    consumer_secret = os.environ['API_SECRET']
    access_token = os.environ['ACCESS_TOKEN']
    access_secret_token = os.environ['ACCESS_TOKEN_SECRET']

    v1_auth = tweepy.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret)
    v1_auth.set_access_token(access_token, access_secret_token)
    v1_api = tweepy.API(v1_auth)

    all_media_ids = []
    for image_path in images:
        media = v1_api.simple_upload(image_path)
        all_media_ids.append(media.media_id)

    post_result = api.create_tweet(
        text=tweet,
        media_ids=all_media_ids
    )

    return post_result


def reply_images(api: tweepy.Client, images,tweet_id):
    """Upload image to Twitter with a tweet"""
    consumer_key = os.environ['API_KEY']
    consumer_secret = os.environ['API_SECRET']
    access_token = os.environ['ACCESS_TOKEN']
    access_secret_token = os.environ['ACCESS_TOKEN_SECRET']

    v1_auth = tweepy.OAuthHandler(consumer_key=consumer_key, consumer_secret=consumer_secret)
    v1_auth.set_access_token(access_token, access_secret_token)
    v1_api = tweepy.API(v1_auth)

    all_media_ids = []
    for image_path in images:
        media = v1_api.simple_upload(image_path)
        all_media_ids.append(media.media_id)

    post_result = api.create_tweet(
        in_reply_to_tweet_id=tweet_id,
        media_ids=all_media_ids
    )

    return post_result

api = verify_twitter_credentials()


match_name = get_match_name(Fotmob_matchID)


client = storage.Client()

bucket_name = "postmatch-dashboards"
folder_prefix_players = f'figures/{today}/players/'
folder_prefix_team = f'figures/{today}/team/'

bucket = client.get_bucket(bucket_name)

if not os.path.exists('figures/players'):
    os.makedirs('figures/players')

if not os.path.exists('figures/team'):
    os.makedirs('figures/team')

# Download player images from Google Cloud Storage
blob_list_players = bucket.list_blobs(prefix=folder_prefix_players)
blob_list_team = bucket.list_blobs(prefix=folder_prefix_team)

player_files = []
team_files = []

# Download player images
for blob in blob_list_players:
    if not blob.name.endswith('/'):
        file_name = os.path.basename(blob.name)
        local_file_path = os.path.join('figures', 'players', file_name)
        blob.download_to_filename(local_file_path)
        player_files.append(local_file_path)

# Download team images
for blob in blob_list_team:
    if not blob.name.endswith('/'):
        file_name = os.path.basename(blob.name)
        local_file_path = os.path.join('figures', 'team', file_name)
        blob.download_to_filename(local_file_path)
        team_files.append(local_file_path)




figures_players_folder = "figures/players"
image_files = os.listdir(figures_players_folder)
image_groups = []

for i in range(0, len(image_files), 4):
    group = image_files[i:i+4]
    image_groups.append(group)

figures = []
for group in image_groups:
    group_paths = [f"{figures_players_folder}/{image}" for image in group]
    figures.append(group_paths)

first_tweet = tweet_images(api, figures[0], tweet= f'{match_name} Players Dashboards')
first_tweet_id = first_tweet.data.id

previous_reply_id = first_tweet_id
for images in figures[1:]:
    reply_to_previous_reply = reply_images(api, images, tweet_id=previous_reply_id)
    previous_reply_id = reply_to_previous_reply.data.id


figures_team_folder = "figures/team"
image_team_files = os.listdir(figures_team_folder)
image_team_groups = []

for i in range(0, len(image_team_files), 4):
    group = image_team_files[i:i+4]
    image_team_groups.append(group)

figures_team = []
for group in image_team_groups:
    group_paths = [f"{figures_team_folder}/{image}" for image in group]
    figures_team.append(group_paths)

first_tweet = tweet_images(api, figures_team[0], tweet= f'{match_name} Dashboards')
first_tweet_id = first_tweet.data.id

previous_reply_id = first_tweet_id
for images in figures_team[1:]:
    reply_to_previous_reply = reply_images(api, images, tweet_id=previous_reply_id)
    previous_reply_id = reply_to_previous_reply.data.id
