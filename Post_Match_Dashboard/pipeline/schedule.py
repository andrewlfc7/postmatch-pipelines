import requests
import json
from pandas import json_normalize
from datetime import datetime

def get_match_date_and_id(team_id):
    # Get data from the API
    response = requests.get(f'https://www.fotmob.com/api/teams?id={team_id}&ccode3=USA_MA')
    data = json.loads(response.content)

    # Extract fixtures data
    fixtures = data['fixtures']['allFixtures']['fixtures']

    # Flatten the JSON and create a DataFrame
    df = json_normalize(fixtures)

    # Get today's date in the format: 'YYYY-MM-DD'
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Check if there is a match today for the specified team
    today_match = df[(df['status.finished'] == True) & (df['status.utcTime'].str.startswith(today_date))]

    if not today_match.empty:
        match_date_str = today_match['status.utcTime'].iloc[0]
        match_date = datetime.strptime(match_date_str, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%d-%m-%y')

        match_id = today_match['id'].iloc[0]

        return f'{match_date}', match_id
    else:
        return None, None

