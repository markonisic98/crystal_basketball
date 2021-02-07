import pandas as pd
import numpy as np

# Helper function to create a unique game ID using a string including the date and two teams involved in a game
# sorted alphabetically
# Takes as argument a cleaned dataframe of the season gamelog of a certain team
def create_game_ID(df):
    array = np.array(df[['TEAM', 'DATE', 'OPP']])
    str_array = [None]*len(array)
    for i in range(len(array)):
        array[i].sort() #sorts alphabetically the list to keep consistent between home/away references
        str_array[i] = ''.join(array[i]) #converts array to string
    return str_array

# Takes as argument a cleaned dataframe of the season gamelog of a certain team
# Cumulative wins at each data point, before the current game has occured - helper function for win 
# percentage moving avg
def cum_win_loss(df):
    cum_win_loss = [None]*df.shape[0] 
    for i in range(df.shape[0]):
        if i == 0: 
            cum_win_loss[i] = 0
        else:
            if df['WIN/LOSS'][i-1] == 'W':
                cum_win_loss[i] = cum_win_loss[i-1] + 1
            else:
                cum_win_loss[i] = cum_win_loss[i-1]
    return cum_win_loss

# Function to return column for win percentage for different spans of time (no. of games previous) for a team
# If a team won 3 of the last 6 games previous to a certain game, their win % would be 0.500. Returns array
# This is like the gen_moving_avg function but in this special case of win%.
def win_percentage_moving_avg(df, len_moving_avg):
    array = cum_win_loss(df)
    moving_avg_array = [0]*len(array)
    for j in range(len(array)):
        if j==0:
            moving_avg_array[j] = 0
        else: 
            if j < len_moving_avg:
                moving_avg_array[j] = (array[j] - array[0])/j
            else:
                moving_avg_array[j] = (array[j] - array[j-len_moving_avg])/len_moving_avg  
    return moving_avg_array

# General moving average calculator for an array lagging 1 period (current period not included) 
# Use length of column or dataframe shape to calculate cumulative average of entire array
def gen_moving_avg(array, len_moving_avg):
    cum_avg_array = [0]*len(array)
    for i in range(len(array)):
        if i==0:
            cum_avg_array[i] = 0
        else: 
            j=max(0, i - len_moving_avg)
            len_avg = min(i-j, len_moving_avg)
            holder_var = 0
            for j in range(j, len_avg+j):
                holder_var = holder_var + array[j]
            cum_avg_array[i] = holder_var/len_avg   
    return cum_avg_array

# Function to find the x game - previous to current game - moving averages for each stat, as well as 
# cumulative total average 
# Takes in list of moving averages user wishes to use: (e.g): [1,3,5], will give moving averages of last 
# 1, 3, and 5 games
# Returns a new dataframe with the addition of columns including the rolling (moving) averages of each 
# column in the original dataframe
def multiple_moving_averages(df, list_moving_avgs):
    new_df = df
    for i in range(df.columns.get_loc("TEAM_PTS"), df.shape[1]):
        for j in range(len(list_moving_avgs)):
            temp_avg = gen_moving_avg(pd.to_numeric(df.iloc[:,i]),list_moving_avgs[j])
            new_df.insert(new_df.shape[1], f"{new_df.columns[i]}_{list_moving_avgs[j]}G_MOVING_AVG", temp_avg)
        temp_avg = gen_moving_avg(pd.to_numeric(df.iloc[:,i]),df.shape[0])
        new_df.insert(new_df.shape[1], f"{new_df.columns[i]}_CUMULATIVE_AVG", temp_avg)
    #special case for win percentage column
    for j in range(len(list_moving_avgs)):
        temp_avg = win_percentage_moving_avg(new_df, list_moving_avgs[j])
        new_df.insert(new_df.shape[1], f"WIN_PERCENTAGE_{list_moving_avgs[j]}G_AVG", temp_avg)
    temp_avg = win_percentage_moving_avg(new_df, new_df.shape[0])
    new_df.insert(new_df.shape[1], "WIN_PERCENTAGE_CUM_AVG", temp_avg)
    
    return new_df

# Function to find the previous record between two teams to be added to the final league-wide dataset
# If I were to assign win percentages to each set of teams in their previous matchups that would mean a 
# team that lost all of its matchups to another team would have the same win percentage as a team that 
# has not played another team yet (0%).
# To account for this, I use a scoring system that starts at 0, and for each game that you have played 
# against a team, it will add 1 for a win, and subtract one for a loss. This means that a team with a 2-2 
# record with another team has the same previous matchup record value as a team that has not played 
# another team yet. If a team is 2-1 vs. another, they will have a value of 1 in this column
# Returns an array to be used as a column in the final dataset creation function.
def calc_matchup_record(df):
    matchup_record_array = [0]*df.shape[0]
    for i in range(df.shape[0]-1, 0, -1): 
        points_holder = 0
        game_ID = df.index[i]
        ref_home_team = df["TEAM"][i]
        ref_away_team = df["OPP"][i] 
        for j in range(i-1, -1, -1):
            home_team = df["TEAM"][j] 
            away_team = df["OPP"][j] 
            # Make 2 strings for each orientation depending on alphabetical order
            substring1 = home_team + away_team
            substring2 = away_team + home_team 
            if substring1 in game_ID or substring2 in game_ID:
                if df["WIN/LOSS"][j] == 'W' and df["TEAM"][j] == ref_home_team: 
                    points_holder += 1
                elif df["WIN/LOSS"][j] == 'L' and df["TEAM"][j] == ref_home_team:
                    points_holder -= 1
                elif df["WIN/LOSS"][j] == 'W' and df["OPP"][j] == ref_home_team:
                    points_holder -= 1
                elif df["WIN/LOSS"][j] == 'L' and df["OPP"][j] == ref_home_team:
                    points_holder += 1
        matchup_record_array[i] = points_holder
        
    return matchup_record_array

# Calculating the home_team actual simple spread in each game and assigning it to a column 
# Also calculating the total game points and assigning it to a column, so the over/under can be compared
# Setting up the odds in decimal form
# Away_Team_Points has been updated to be called away_team_pts in the OG dataset creator
# Takes as input a dataframe created in this project
def calc_combined_pts_and_decimal_odds(df):
    actual_home_spread = []
    total_pts = []
    home_odds = []
    away_odds = []
    # Combined points and decimal odds calculation
    for i, (pts1, pts2, odds1, odds2) in enumerate(zip(df['HOME_TEAM_PTS'], df['AWAY_TEAM_PTS'], 
                                                       df['TEAM_ODDS'], df['OPP_ODDS'])):
        actual_home_spread.append(pts2-pts1)
        total_pts.append(pts1+pts2)
        if odds1 < 0 and odds2 < 0: # If odds are relatively even plus house cut
            home_odds.append((odds1-100)/odds1)
            away_odds.append((odds2-100)/odds2)
        elif odds1 < 0:
            home_odds.append((odds1-100)/odds1)
            away_odds.append((odds2+100)/100)
        else:
            # Dynasties against scrubs won't even get odds occasionally, so check for 0 odds
            # And assign the team with the point spread odds of -50000
            # Also bookies scared to set odds sometimes because they're confused???
            if (odds1 == 0 or odds2 == 0) and df['TEAM_SPREAD'][i] < 0: # Home team dynasty
                odds1 = -1000
                odds2 = 1000
            elif (odds1 == 0 or odds2 == 0) and df['TEAM_SPREAD'][i] > 0: # Home team scrubs
                odds1 = 1000
                odds2 = -1000
            elif (odds1 == 0 or odds2 == 0) and df['TEAM_SPREAD'][i] == 0: # Sportsbooks scared :(
                odds1 = -110
                odds2 = -110
                       
            home_odds.append((odds1+100)/100)
            away_odds.append((odds2-100)/odds2)

    return total_pts, actual_home_spread, home_odds, away_odds

