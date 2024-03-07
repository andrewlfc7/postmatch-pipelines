import datetime

from Football_Analysis_Tools import whoscored_data_engineering, whoscored_custom_events
import pandas as pd
import numpy as np
import json


def data_processing(data):
    away_postion = pd.DataFrame(data['matchCentreData']['away']['players'])

    home_postion = pd.DataFrame(data['matchCentreData']['home']['players'])

    away_postion = away_postion[['playerId', 'position', 'shirtNo']]

    home_postion = home_postion[['playerId', 'position', 'shirtNo']]

    event_data = data['matchCentreData']

    player_id = event_data['playerIdNameDictionary']

    match_id = data['matchId']

    home_data = data['matchCentreData']['home']

    home_team_name = event_data['home']['name']

    away_team_name = event_data['away']['name']

    home_teamID = event_data['home']['teamId']

    away_teamID = event_data['away']['teamId']

    events = event_data['events']

    score = data['matchCentreData']['score']
    match_date_str = datetime.datetime.strptime(data['matchCentreData']['startDate'], '%Y-%m-%dT%H:%M:%S')
    match_date = match_date_str.date()
    match_date = match_date.strftime('%Y-%m-%d')



    events = pd.DataFrame(events)

    events['Score'] = score
    events['match_date'] = match_date

    events['match_id'] = match_id

    match_string = home_team_name + " " + "-" + " " + away_team_name

    events['match_string'] = match_string

    # Create a dictionary mapping team IDs to team names
    team_name = {
        home_teamID: home_team_name,
        away_teamID: away_team_name
    }

    events['team_name'] = events['teamId'].map(team_name)

    # Create a dictionary mapping team IDs to team names
    team_dict = {home_teamID: 'Home', away_teamID: 'Away'}

    # Add a column to the event data to indicate whether the corresponding data is for the home team or away team
    events['Venue'] = events['teamId'].map(team_dict)


    # Assume that the events df has a column named 'playerId'
    # and home/away df has columns named 'playerId', 'position', 'shirtno'

    def get_player_info(row):
        # check if the venue is home or away
        if row['Venue'] == 'Home':
            player_info = home_postion.loc[home_postion['playerId'] == row['playerId'], ['position', 'shirtNo']]
        else:
            player_info = away_postion.loc[away_postion['playerId'] == row['playerId'], ['position', 'shirtNo']]

        # if the playerId is not found in home/away df, set position and shirtno to NaN
        if player_info.empty:
            return pd.Series({'position': np.nan, 'shirtNo': np.nan})
        else:
            return player_info.squeeze()


    # apply the function to each row of the events df
    events[['position', 'shirtNo']] = events.apply(get_player_info, axis=1)

    # Load the event ID json file
    with open('Post_Match_Dashboard/pipeline/data/event_id.json') as f:
        event_id = {value: key for key, value in json.load(f).items()}

    # Replace the values in the 'satisfiedEventsTypes' column with the corresponding event type labels
    events['satisfiedEventsTypes'] = events['satisfiedEventsTypes'].apply(lambda x: [event_id.get(i) for i in x])

    # Extract the outcomeType values using the apply() function and a lambda function
    events['outcomeType'] = events['outcomeType'].apply(lambda x: x['displayName'])

    events['type'] = events['type'].apply(lambda x: x['displayName'])

    events['period'] = events['period'].apply(lambda x: x['displayName'])

    events['qualifiers'] = events['qualifiers'].apply(
        lambda x: [{item['type']['displayName']: item.get('value', True)} for item in x])

    events = events.rename(columns={'type': 'event_type'})

    events['period'] = events['period'].map({'FirstHalf': 1, 'SecondHalf': 2})

    events = whoscored_data_engineering.cumulative_match_mins(events)

    events = whoscored_custom_events.get_xthreat(events)

    for i, row in events.iterrows():
        player_id_val = row['playerId']
        if not pd.isnull(player_id_val):
            player_name = player_id.get(str(int(player_id_val)), 'Unknown')
            events.at[i, 'playerName'] = player_name
        else:
            events.at[i, 'playerName'] = 'Unknown'



    def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=60, min_carry_duration=1,
                            max_carry_duration=10):
        """ Add carry events to whoscored-style events dataframe
        Function to read a whoscored-style events dataframe (single or multiple matches) and return an event dataframe
        that contains carry information.
        Args:
            events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
            min_carry_length (float, optional): minimum distance required for event to qualify as carry. 5m by default.
            max_carry_length (float, optional): largest distance in which event can qualify as carry. 60m by default.
            min_carry_duration (float, optional): minimum duration required for event to quality as carry. 2s by default.
            max_carry_duration (float, optional): longest duration in which event can qualify as carry. 10s by default.
        Returns:
            pandas.DataFrame: whoscored-style dataframe of events including carries
        """

        # Initialise output dataframe
        events_out = pd.DataFrame()

        # Carry conditions (convert from metres to opta)
        min_carry_length = 3.0
        max_carry_length = 60.0
        min_carry_duration = 1.0
        max_carry_duration = 10.0

        for match_id in events_df['match_id'].unique():

            match_events = events_df[events_df['match_id'] == match_id].reset_index()
            match_carries = pd.DataFrame()

            for idx, match_event in match_events.iterrows():

                if idx < len(match_events) - 1:
                    prev_evt_team = match_event['teamId']
                    next_evt_idx = idx + 1
                    init_next_evt = match_events.loc[next_evt_idx]
                    take_ons = 0
                    incorrect_next_evt = True

                    while incorrect_next_evt:

                        next_evt = match_events.loc[next_evt_idx]

                        if next_evt['event_type'] == 'TakeOn' and next_evt['outcomeType'] == 'Successful':
                            take_ons += 1
                            incorrect_next_evt = True

                        elif ((next_evt['event_type'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful')
                              or (next_evt['teamId'] != prev_evt_team and next_evt['event_type'] == 'Challenge' and
                                  next_evt[
                                      'outcomeType'] == 'Unsuccessful')
                              or (next_evt['event_type'] == 'Foul')):
                            incorrect_next_evt = True

                        else:
                            incorrect_next_evt = False

                        next_evt_idx += 1

                    # Apply some conditioning to determine whether carry criteria is satisfied

                    same_team = prev_evt_team == next_evt['teamId']
                    not_ball_touch = match_event['event_type'] != 'BallTouch'
                    dx = 105 * (match_event['endX'] - next_evt['x']) / 100
                    dy = 68 * (match_event['endY'] - next_evt['y']) / 100
                    far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
                    not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
                    dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
                    min_time = dt >= min_carry_duration
                    same_phase = dt < max_carry_duration
                    same_period = match_event['period'] == next_evt['period']

                    valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase & same_period

                    if valid_carry:
                        carry = pd.DataFrame()
                        prev = match_event
                        nex = next_evt

                        carry.loc[0, 'eventId'] = prev['eventId'] + 0.5
                        carry['minute'] = np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                                prev['minute'] * 60 + prev['second'])) / (2 * 60))
                        carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) +
                                            (prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                        carry['teamId'] = nex['teamId']
                        carry['x'] = prev['endX']
                        carry['y'] = prev['endY']
                        carry['expandedMinute'] = np.floor(
                            ((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) +
                             (prev['expandedMinute'] * 60 + prev['second'])) / (2 * 60))
                        carry['period'] = nex['period']
                        carry['type'] = carry.apply(lambda x: {'value': 99, 'displayName': 'Carry'}, axis=1)
                        carry['outcomeType'] = 'Successful'
                        carry['qualifiers'] = carry.apply(
                            lambda x: {'type': {'value': 999, 'displayName': 'takeOns'}, 'value': str(take_ons)},
                            axis=1)
                        carry['satisfiedEventsTypes'] = carry.apply(lambda x: [], axis=1)
                        carry['isTouch'] = True
                        carry['playerName'] = nex['playerName']
                        carry['endX'] = nex['x']
                        carry['endY'] = nex['y']
                        carry['blockedX'] = np.nan
                        carry['blockedY'] = np.nan
                        carry['goalMouthZ'] = np.nan
                        carry['goalMouthY'] = np.nan
                        carry['isShot'] = np.nan
                        carry['relatedEventId'] = nex['eventId']
                        carry['relatedPlayerId'] = np.nan
                        carry['isGoal'] = np.nan
                        carry['cardType'] = np.nan
                        carry['isOwnGoal'] = np.nan
                        carry['match_id'] = nex['match_id']
                        carry['match_date'] = nex['match_date']
                        carry['position'] = nex['position']
                        carry['playerId'] = nex['playerId']
                        carry['Venue'] = nex['Venue']
                        carry['xThreat'] = nex['xThreat']
                        carry['xThreat_gen'] = nex['xThreat_gen']
                        carry['event_type'] = 'Carry'
                        carry['cumulative_mins'] = (prev['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2

                        match_carries = pd.concat([match_carries, carry], ignore_index=True, sort=False)

            match_events_and_carries = pd.concat([match_carries, match_events], ignore_index=True, sort=False)
            match_events_and_carries = match_events_and_carries.sort_values(
                ['match_id', 'period', 'cumulative_mins']).reset_index(drop=True)

            # Rebuild events dataframe
            events_out = pd.concat([events_out, match_events_and_carries])

        return events_out


    events = insert_ball_carries(events)

    return events


# data['qualifiers'] = [literal_eval(x) for x in data['qualifiers']]
# data['satisfiedEventsTypes'] = [literal_eval(x) for x in data['satisfiedEventsTypes']]
def custom_events(data):
    data.loc[:, 'is_open_play'] = True
    for index, record in enumerate(data['qualifiers']):
        for attr in record:
            if isinstance(attr, dict):
                key_search = list(attr.keys())[0]
                if key_search in ['GoalKick', 'FreekickTaken', 'CornerTaken', 'PenaltyFaced', 'CornerAwarded', 'ThrowIn',
                                  'OffsideGiven', 'foulGiven']:
                    data.at[index, 'is_open_play'] = False


    def check_if_pass_is_progressive(x, y, end_x, end_y):
        '''
        This function returns "True" if the pass meets the criteria
        for a progressive pass.
        '''
        # -- Start position
        height_start = abs(x - 100)
        length_start = abs(y - 50)
        distance_sq_start = height_start ** 2 + length_start ** 2
        distance_start = distance_sq_start ** (1 / 2)
        # -- End position
        height_end = abs(end_x - 100)
        length_end = abs(end_y - 50)
        distance_sq_end = height_end ** 2 + length_end ** 2
        distance_end = distance_sq_end ** (1 / 2)
        # -- Calculate change in distance
        delta_distance = distance_end / distance_start - 1
        if delta_distance <= -0.25:
            return True
        else:
            return False


    data['is_progressive'] = data.apply(lambda x: check_if_pass_is_progressive(x['x'], x['y'], x['endX'], x['endY']),
                                        axis=1)


    def check_if_pass_is_into_box(x, y, end_x, end_y):
        '''
        This function returns "True" if the pass meets the criteria
        for a progressive pass and is successful into the box.
        '''
        # -- Start position
        height_start = abs(x - 100)
        length_start = abs(y - 50)
        distance_sq_start = height_start ** 2 + length_start ** 2
        distance_start = distance_sq_start ** (1 / 2)
        # -- End position
        height_end = abs(end_x - 100)
        length_end = abs(end_y - 50)
        distance_sq_end = height_end ** 2 + length_end ** 2
        distance_end = distance_sq_end ** (1 / 2)
        # -- Calculate change in distance
        delta_distance = distance_end / distance_start - 1
        # -- Determine pass end position and whether it's a successful pass into the box
        x_position = 120 * end_x / 100
        y_position = 80 * end_y / 100
        if delta_distance <= -0.25 and x_position >= 102 and 18 <= y_position <= 62:
            return True
        else:
            return False


    data['is_pass_into_box'] = data.apply(lambda x: check_if_pass_is_into_box(x['x'], x['y'], x['endX'], x['endY']), axis=1)

    recovery_set = {'ballRecovery', 'interceptionWon', 'tackleWon', 'foulGiven', 'duelAerialWon'}
    data = data.copy()
    data['won_possession'] = False
    for index, row in enumerate(data['satisfiedEventsTypes']):
        set_element = set(row)
        if len(recovery_set.intersection(set_element)) > 0:
            data.at[index, 'won_possession'] = True
    chances_set = {'keyPassLong', 'keyPassShort', 'keyPassCross', 'keyPassCorner', 'keyPassThroughball',
                   'keyPassFreekick', 'bigChanceCreated', 'passkey'}
    data = data.copy()
    data['key_pass'] = False
    for index, row in enumerate(data['satisfiedEventsTypes']):
        set_element = set(row)
        if len(chances_set.intersection(set_element)) > 0:
            data.at[index, 'key_pass'] = True

    assist_set = {'assistCross', 'assistCorner', 'assistThroughball', 'intentionalAssist', 'assistFreekick',
                  'assistThrowin', 'assistOther', 'assist'}
    data = data.copy()
    data['assist'] = False
    for index, row in enumerate(data['satisfiedEventsTypes']):
        set_element = set(row)
        if len(assist_set.intersection(set_element)) > 0:
            data.at[index, 'assist'] = True

    FinalThirdPasses_set = {'successfulFinalThirdPasses'}
    data = data.copy()
    data['FinalThirdPasses'] = False
    for index, row in enumerate(data['satisfiedEventsTypes']):
        set_element = set(row)
        if len(FinalThirdPasses_set.intersection(set_element)) > 0:
            data.at[index, 'FinalThirdPasses'] = True


    # turnover = {'turnover'}
    # data = data.copy()
    # data['turnover'] = False
    # for index, row in enumerate(data['satisfiedEventsTypes']):
    #     set_element = set(row)
    #     if len(turnover.intersection(set_element)) > 0:
    #         data.at[index, 'turnover'] = True
    #
    #
    #
    # defensiveThird = {'defensiveThird'}
    # data = data.copy()
    # data['defensiveThird'] = False
    # for index, row in enumerate(data['satisfiedEventsTypes']):
    #     set_element = set(row)
    #     if len(defensiveThird.intersection(set_element)) > 0:
    #         data.at[index, 'defensiveThird'] = True
    #


    # midThird = {'midThird'}
    # data = data.copy()
    # data['midThird'] = False
    # for index, row in enumerate(data['satisfiedEventsTypes']):
    #     set_element = set(row)
    #     if len(midThird.intersection(set_element)) > 0:
    #         data.at[index, 'midThird'] = True


    # finalThird = {'finalThird'}
    # data = data.copy()
    # data['finalThird'] = False
    # for index, row in enumerate(data['satisfiedEventsTypes']):
    #     set_element = set(row)
    #     if len(finalThird.intersection(set_element)) > 0:
    #         data.at[index, 'finalThird'] = True



    data['is_carry_into_box'] = data.apply(whoscored_custom_events.carry_into_box, axis=1)

    data['progressive_carry'] = data.apply(whoscored_custom_events.progressive_carry, axis=1)
    # Get the passes dataframe
    # data = data[data["event_type"] == 'Pass']

    # Create a boolean mask for assist events
    assist_mask = data["assist"] == True

    # Initialize pre-assist list with False values
    pre_assist_list = [False] * len(data)

    # Loop through each assist event
    for idx, row in data[assist_mask].iterrows():
        assist_idx = data.index.get_loc(idx)

        pre_assist = data.iloc[assist_idx - 1]
        if pre_assist["event_type"] == "Pass" and pre_assist["playerName"] != row["playerName"]:
            pre_assist_list[assist_idx - 1] = True

    # Add the pre-assist list as a new column in the dataframe
    data['pre_assist'] = pre_assist_list

    data.update(data[['pre_assist']])



    def check_if_pass_is_switch(row):
        # -- Start position
        height_start = abs(row['x'] - 100)
        length_start = abs(row['y'] - 50)
        distance_sq_start = height_start ** 2 + length_start ** 2
        distance_start = distance_sq_start ** (1 / 2)
        # -- End position
        height_end = abs(row['endX'] - 100)
        length_end = abs(row['endY'] - 50)
        distance_sq_end = height_end ** 2 + length_end ** 2
        distance_end = distance_sq_end ** (1 / 2)
        # -- Calculate change in distance
        delta_distance = distance_end / distance_start - 1
        if delta_distance <= -0.25:
            return True
        else:
            return False


    # Calculate distance and angle
    data['distance'] = np.sqrt((data['endX'] - data['x']) ** 2 + (data['endY'] - data['y']) ** 2)
    data['angle'] = np.arctan2(data['endY'] - data['y'], data['endX'] - data['x'])

    # Identify switches
    data['switch'] = data.apply(check_if_pass_is_switch, axis=1)
    data.loc[data['distance'] < 40, 'switch'] = False

    return data




