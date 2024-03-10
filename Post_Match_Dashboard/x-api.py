import os
from dotenv import load_dotenv
import tweepy
import datetime
import pytz



eastern = pytz.timezone('US/Eastern')
today = datetime.datetime.now(eastern).date()
today = today.strftime('%Y-%m-%d')


load_dotenv()


def verify_twitter_credentials():
    """Verify twitter authentication"""
    consumer_key = os.getenv('API_KEY')
    consumer_secret = os.getenv('API_SECRET')
    access_token = os.getenv('ACCESS_TOKEN')
    access_secret_token = os.getenv('ACCESS_TOKEN_SECRET')

    api = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_secret_token
    )

    return api

def tweet_images(api: tweepy.Client, images, tweet=''):
    """Upload image to Twitter with a tweet"""
    consumer_key = os.getenv('API_KEY')
    consumer_secret = os.getenv('API_SECRET')
    access_token = os.getenv('ACCESS_TOKEN')
    access_secret_token = os.getenv('ACCESS_TOKEN_SECRET')


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
    consumer_key = os.getenv('API_KEY')
    consumer_secret = os.getenv('API_SECRET')
    access_token = os.getenv('ACCESS_TOKEN')
    access_secret_token = os.getenv('ACCESS_TOKEN_SECRET')


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




# figures = [
#     f'figures/match_avgDashboard{today}.png',
#     f'figures/Alexis_Mac_Allister_avgDashboard{today}.png',
#     f'figures/Cody_Gakpo_avgDashboard{today}.png',
#     f'figures/Conor_Bradley_avgDashboard{today}.png'
# ]
#

figures = [
    # 'figures/dashboardmain2024-03-02.png',
    # 'figures/dashboard_odds2024-03-02.png',
    'figures/dashboard_xT_heatmap2024-03-02.png',
    # 'figures/match_avgDashboard2024-03-02.png'
]

figures_v1 = [
    'figures/Dominik_Szoboszlai_avgDashboard2024-03-02.png'
]

# api = verify_twitter_credentials()
# tweet_result = tweet_images(api, figures, tweet='testing')
# print("Tweet posted successfully:", tweet_result)


# tweet_id = tweet_result.data['id']

# reply_images(api, figures_v1,tweet_result.data['id'])


# Get images from the 'figures' folder
figure_folder = 'figures'
figure_files = os.listdir(figure_folder)

# Separate images into groups of four for each tweet
for i in range(0, len(figure_files), 4):
    images = [os.path.join(figure_folder, file) for file in figure_files[i:i+4]]
    print(images)





# import os
# from dotenv import load_dotenv
# import tweepy
# import datetime
# import pytz
#
# eastern = pytz.timezone('US/Eastern')
# today = datetime.datetime.now(eastern).date()
# today = today.strftime('%Y-%m-%d')
#
# load_dotenv()
#
# def verify_twitter_credentials():
#     """Verify twitter authentication"""
#     consumer_key = os.getenv('API_KEY')
#     consumer_secret = os.getenv('API_SECRET')
#     access_token = os.getenv('ACCESS_TOKEN')
#     access_secret_token = os.getenv('ACCESS_TOKEN_SECRET')
#
#     api = tweepy.Client(
#         consumer_key=consumer_key,
#         consumer_secret=consumer_secret,
#         access_token=access_token,
#         access_token_secret=access_secret_token
#     )
#
#     return api
#
# def tweet_images(api: tweepy.Client, images, tweet=''):
#     """Upload images to Twitter with a tweet"""
#     all_media_ids = []
#     for image_path in images:
#         media = api.simple_upload(image_path)
#         all_media_ids.append(media.media_id)
#
#     post_result = api.create_tweet(
#         text=tweet,
#         media_ids=all_media_ids
#     )
#
#     return post_result
#
# def reply_to_tweet_with_image(api: tweepy.Client, image_path, tweet_id, reply_text=''):
#     """Reply to the original tweet with an image"""
#     media = api.simple_upload(image_path)
#     media_id = media.media_id
#
#     post_result = api.create_tweet(
#         text=reply_text,
#         in_reply_to_tweet_id=tweet_id,
#         media_ids=[media_id]
#     )
#
#     return post_result
#
# # Example usage:
# api = verify_twitter_credentials()
#
# # 1. Tweeting with multiple images
# figures = [
#     'figures/dashboardmain2024-03-02.png',
#     'figures/dashboard_odds2024-03-02.png',
#     'figures/dashboard_xT_heatmap2024-03-02.png',
#     'figures/match_avgDashboard2024-03-02.png'
# ]
# tweet_result = tweet_images(api, figures, tweet='Testing multiple images')
# print("Tweet posted successfully:", tweet_result)
#
# # 2. Replying to a tweet with an image
# reply_image_path = 'figures/dashboardmain2024-03-02.png'
# reply_tweet_id = '1766578688609464578'  # Replace with the original tweet ID
# reply_result = reply_to_tweet_with_image(api, reply_image_path, reply_tweet_id, reply_text='Replying with an image')
# print("Reply posted successfully:", reply_result)
