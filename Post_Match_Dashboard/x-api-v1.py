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

if not os.path.exists('figures'):
    os.makedirs('figures')


blob_list_players = bucket.list_blobs(prefix=folder_prefix_players)
player_files = []
for blob in blob_list_players:
    if not blob.name.endswith('/'):
        file_name = blob.name.split('/')[-1]
        blob.download_to_filename(f'figures/{file_name}')
        player_files.append(f'figures/{file_name}')

player_images_grouped = [player_files[i:i+4] for i in range(0, len(player_files), 4)]

print(player_images_grouped)

for i, player_images in enumerate(player_images_grouped):
    if player_images:
        tweet_content = f'{match_name} Players Dashboards'
        tweet_result = tweet_images(api, player_images, tweet=tweet_content)
        print("Player main tweet posted successfully:", tweet_result)
        player_first_tweet_id = tweet_result.data['id']
        player_other_images = player_images[1:]
        for image in player_other_images:
            reply_result = reply_images(api, [image], player_first_tweet_id)
            print("Player dashboard reply posted successfully:", reply_result)
