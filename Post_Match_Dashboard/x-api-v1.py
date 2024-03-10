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

# Define bucket name and folder prefix
bucket_name = "postmatch-dashboards"
folder_prefix = f'figures/{today}/'

# Get the bucket
bucket = client.get_bucket(bucket_name)

# List blobs with the specified folder prefix
blob_list = bucket.list_blobs(prefix=folder_prefix)

# Extract figure files from blob names
figure_files = [blob.name.split('/')[-1] for blob in blob_list if not blob.name.endswith('/')]

# Extract player and team files
player_files = [file for file in figure_files if 'players' in file]
team_files = [file for file in figure_files if 'team' in file]

# Tweet player images
if player_files:
    for i in range(0, len(player_files), 4):
        player_images = player_files[i:i+4]
        if player_images:
            tweet_result = tweet_images(api, player_images, tweet=f'{match_name} Players Dashboards')
            print("Player main tweet posted successfully:", tweet_result)
            player_first_tweet_id = tweet_result.data['id']

            player_other_images = player_images[1:]

            for image in player_other_images:
                reply_result = reply_images(api, [image], player_first_tweet_id)
                print("Player dashboard reply posted successfully:", reply_result)

# Tweet team images
team_main_images = [os.path.join(folder_prefix, file) for file in team_files if 'main' in file]
team_other_images = [os.path.join(folder_prefix, file) for file in team_files if 'main' not in file]

if team_main_images:
    for i in range(0, len(team_main_images), 4):
        team_main_images_batch = team_main_images[i:i+4]
        tweet_result = tweet_images(api, team_main_images_batch, tweet=f'{match_name} Team Dashboards')
        print("Team main tweet posted successfully:", tweet_result)
        team_main_tweet_id = tweet_result.data['id']

        other_images_batch = team_other_images[i:i+4]
        for image in other_images_batch:
            reply_result = reply_images(api, [image], team_main_tweet_id)
            print("Team main reply posted successfully:", reply_result)
