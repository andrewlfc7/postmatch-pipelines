import json
import requests
import pandas as pd
def get_shots_data(match_id):
    response = requests.get(f'https://www.fotmob.com/api/matchDetails?matchId={match_id}&ccode3=USA&timezone=America%2FChicago&refresh=true&includeBuzzTab=false&acceptLanguage=en-US')

    data = response.content
    data = json.loads(data)

    matchId = data['general']['matchId']
    matchTimeUTCDate = data['general']['matchTimeUTCDate'][:10]

    competitions = data['general']['parentLeagueName']
    teamcolors = data['general']['teamColors']


    homeTeam = data['general']['homeTeam']
    awayTeam = data['general']['awayTeam']

    home_team_id = homeTeam['id']
    away_team_id = awayTeam['id']

    homeTeamName = homeTeam['name']
    awayTeamName = awayTeam['name']

    homeTeam = pd.DataFrame(homeTeam, index=[0])
    awayTeam = pd.DataFrame(awayTeam, index=[0])

    shot_data = data['content']['shotmap']['shots']

    df_shot = pd.DataFrame(shot_data)

    df_shot['match_id'] = matchId
    df_shot['match_date'] = matchTimeUTCDate
    df_shot['competition'] = competitions


    df_shot['Venue'] = ''
    for index, row in df_shot.iterrows():
        if row['teamId'] == home_team_id:
            df_shot.loc[index, 'Venue'] = 'Home'
            df_shot.loc[index, 'TeamName'] = homeTeamName
        elif row['teamId'] == away_team_id:
            df_shot.loc[index, 'Venue'] = 'Away'
            df_shot.loc[index, 'TeamName'] = awayTeamName

    def extract_value(d, key):
        return d[key]

    df_shot['onGoalShot_X'] = df_shot['onGoalShot'].apply(extract_value, args=('x',))
    df_shot['onGoalShot_Y'] = df_shot['onGoalShot'].apply(extract_value, args=('y',))
    df_shot['onGoalShot_ZR'] = df_shot['onGoalShot'].apply(extract_value, args=('zoomRatio',))
    # df_shot.drop(['onGoalShot'], axis=1, inplace=True)
    if 'shortName' in df_shot.columns:
        df_shot.drop(['shortName'], axis=1, inplace=True)

    df_shot = df_shot[[
        'id', 'eventType', 'teamId', 'playerId', 'playerName', 'x', 'y', 'min',
        'minAdded', 'isBlocked', 'isOnTarget', 'blockedX', 'blockedY',
        'goalCrossedY', 'goalCrossedZ', 'expectedGoals',
        'expectedGoalsOnTarget', 'shotType', 'situation', 'period', 'isOwnGoal',
        'isSavedOffLine', 'firstName', 'lastName', 'fullName', 'teamColor',
        'match_id', 'match_date', 'competition', 'Venue', 'TeamName',
        'onGoalShot_X', 'onGoalShot_Y', 'onGoalShot_ZR'

    ]]


    return df_shot
