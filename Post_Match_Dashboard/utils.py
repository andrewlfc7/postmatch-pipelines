import requests
from PIL import Image
from io import BytesIO
import os
import numpy as np


def check_logo_existence(away_id, home_id):
    logo_folder = 'Post_Match_Dashboard/Data/team_logo'

    away_logo_path = f'{logo_folder}Fotmob_{away_id}.png'
    home_logo_path = f'{logo_folder}Fotmob_{home_id}.png'

    return os.path.exists(away_logo_path) and os.path.exists(home_logo_path)

def get_and_save_logo(team_id):
    fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
    response = requests.get(f'{fotmob_url}{team_id:.0f}.png')
    club_icon = Image.open(BytesIO(response.content))

    logo_path = f'Post_Match_Dashboard/Data/team_logo/{team_id}.png'
    club_icon.save(logo_path, 'PNG', quality=95, dpi=(300, 300))

def ax_logo(team_id, ax):
    '''
    Plots the logo of the team at a specific axes.
    Args:
        team_id (int): the id of the team according to Fotmob. You can find it in the url of the team page.
        ax (object): the matplotlib axes where we'll draw the image.
    '''
    logo_folder = 'Post_Match_Dashboard/Data/team_logo/'
    logo_path = f'{logo_folder}{team_id}.png'

    if not os.path.exists(logo_path):
        get_and_save_logo(team_id)

    club_icon = Image.open(logo_path)
    ax.imshow(club_icon)
    ax.axis('off')
    return ax


def mark_turnover_followed_by_shot(df, window_size=8):
    # Initialize a new column to indicate whether turnover was followed by a shot
    df['turnover_followed_by_shot'] = False

    # Iterate through the DataFrame
    for index, row in df.iterrows():
        # Check if the event is a turnover
        if row['turnover']:
            # Iterate through the next events within the specified window size
            for i in range(index + 1, min(index + window_size + 1, len(df))):
                # Check if the event is a shot
                if df.loc[i, 'isShot']:
                    # If a shot is found within the specified window after the turnover, mark it and break the loop
                    df.at[index, 'turnover_followed_by_shot'] = True
                    break

    return df


def pass_angle(x, y, endX, endY):
    vector1 = np.column_stack((endX - x, endY - y))
    vector2 = np.array([1, 0])

    cosTh = np.dot(vector1, vector2) / (np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2))
    sinTh = np.cross(vector1, vector2) / (np.linalg.norm(vector1, axis=1) * np.linalg.norm(vector2))

    angles_rad = np.arctan2(sinTh, cosTh)
    angles_deg = np.rad2deg(angles_rad)

    return angles_deg
