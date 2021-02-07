# Scraping Starting Lineups from the Web
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from dataCollection.helpers import create_game_ID, multiple_moving_averages
from datetime import datetime
from calendar import month_abbr
from dataCollection.teamAbbreviations import TEAM_TO_TEAM_ABBR

# Enter date format as such: YYYY-MM-DD
# Enter team name as 3 letter abbrev
# Returns a list with the team's active lineup for the specific game
def get_game_lineup(team, date): 
    # Handle the difference in bball reference dictionary and other sources
    if team is 'BRK': 
        team = 'BKN' 
    if team is 'CHO':
        team = 'CHA'
    URL = f'https://rotogrinders.com/lineups/nba?date={date}&site=draftkings'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    list_active_players = []
    content = soup.find("li", {"data-role": "lineup-card", "data-home": f"{team}"})
    if content is None:
        content = soup.find("li", {"data-role": "lineup-card", "data-away": f"{team}"})
        content_2 = content.find("div", {"class": "blk away-team"})
        content_3 = content_2.findAll("span", {"class": "pname"})
        for each in content_3:
            list_active_players.append(each.get_text().strip('\n'))
    else:
        content_2 = content.find("div", {"class": "blk home-team"})
        content_3 = content_2.findAll("span", {"class": "pname"})
        for each in content_3:
            list_active_players.append(each.get_text().strip('\n'))

    return list_active_players

# Get the odds of each game 
# Note that this data only goes back to the 2014/2015 season, and Brooklyn abbrev is 'BKN' here instead
# of 'BRK' in  basketball reference.
# Also they only had starting 5s at beginning of 2014/2015 season (only from the new year onward)
# To be safe, avoid using any years of data before the 2015/2016 nba regular season
def get_game_odds(team, date):
    # Handle differences in team abbreviations
    if team is 'BRK': 
        team = 'BKN'
    if team is 'CHO':
        team = 'CHA'
    URL = f'https://rotogrinders.com/lineups/nba?date={date}&site=draftkings'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    content = soup.find("li", {"data-role": "lineup-card", "data-home": f"{team}"})
    if content is None:
        content = soup.find("li", {"data-role": "lineup-card", "data-away": f"{team}"})
        content2 = content.find("div", {"class": "ou"})
        if content2 is None: # In case of error where there are no avail. odds, provide generic odds
            team_implied_pts = '100'
            team_odds = '-110'
            over_under = '200'
            team_spread = '0'
            opp_implied_pts = '100'
            opp_odds = '-110'
        else:               
            string = content2.get_text()
            list = string.split()
            if len(list) !=7:
                team_implied_pts = '100'
                team_odds = '-110'
                over_under = '200'
                team_spread = '0'
                opp_implied_pts = '100'
                opp_odds = '-110'
            else: 
                team_implied_pts = list[0]
                team_odds = list[1].strip('(').strip(')')
                team_odds = team_odds.replace(",","")
                over_under = list[2]
                if list[3] == team: # Checking whether the team in question is home or away
                    team_spread = float(list[4])
                else:
                    team_spread = -(float(list[4]))
                opp_implied_pts = list[5]
                opp_odds = list[6].strip('(').strip(')')
                opp_odds = opp_odds.replace(',','')
    else:
        content2 = content.find("div", {"class": "ou"})
        if content2 is None: # In case of error where there are no avail. odds, provide generic odds
            team_implied_pts = '100'
            team_odds = '-110'
            over_under = '200'
            team_spread = '0'
            opp_implied_pts = '100'
            opp_odds = '-110'
        else:
            string = content2.get_text()
            list = string.split()
            if len(list) !=7:
                team_implied_pts = '100'
                team_odds = '-110'
                over_under = '200'
                team_spread = '0'
                opp_implied_pts = '100'
                opp_odds = '-110'
            else: 
                opp_implied_pts = list[0]
                opp_odds = list[1].strip('(').strip(')')
                opp_odds = opp_odds.replace(',','')
                over_under = list[2]
                if list[3] == team: #change to home team
                    team_spread = float(list[4])
                else:
                    team_spread = -(float(list[4]))
                team_implied_pts = list[5]
                team_odds = list[6].strip('(').strip(')')
                team_odds = team_odds.replace(",","")
    # Create a list with all of the values to be returned
    final_list = [None]*6
    final_list[0] = float(team_implied_pts)
    final_list[1] = float(team_odds)
    final_list[2] = float(over_under)
    final_list[3] = float(team_spread)
    final_list[4] = float(opp_implied_pts)
    final_list[5] = float(opp_odds)
    
    return final_list

# helper function to scrape an upcoming game's info for a team and date, to be used 
# in the get_team_game_log function
# Takes as input team name as an abbreviation, and date of desired game
# Default value of date is today's date
def get_upcoming_game(team, date = datetime.today().strftime('%Y-%m-%d')):
    
    if date < datetime.today().strftime('%Y-%m-%d'):
        print("Upcoming game was not included because it has already occured.\
 Please input a date later than yesterday.")
        return pd.DataFrame() # return empty dataframe
    date_year = date.split('-')[0]
    URL = f"https://www.basketball-reference.com/teams/{team}/{date_year}_start.html"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    schedule = soup.find("table")
    df_schedule = pd.read_html(str(schedule))[0]
    df_schedule.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    df_schedule.rename(columns={"Unnamed: 5": "HOME/AWAY"}, inplace=True)

    # reformat table to fit into get_team_game_log_upcoming columns
    dates=[None]*df_schedule.shape[0]
    for i, date_unformatted in enumerate(df_schedule.Date):
        dates[i] = date_unformatted.split(',')

    # split input date into its separate values, to compare to new format date    
    date_split = date.split('-')
    date_split[1] = int(date_split[1]) # to make the month an int for month_abbr func.
    date_split[2] = int(date_split[2]) #to take care of the zeros before a day date

    # iterate through dates into dataframe to find matching date to input  
    game_info = pd.DataFrame()
    for i, row in enumerate(dates):
        row[1] = row[1].strip(' ')
        row[1] = row[1].split(' ')
        row[1][1] = int(row[1][1]) #to take care of the zeros before a day date 
        row[2] = row[2].strip(' ')

        # check if the input date and table date match, to return that game's info in dataframe
        if (date_split[0] == row[2] and month_abbr[date_split[1]] == row[1][0] and
            date_split[2] == row[1][1]):
            # check which team is home/away
            new_df = df_schedule.iloc[i,:].to_frame()
            game_info = new_df.transpose()
            # create df format acceptable by "create_game_ID" function
            game_info.insert(0, "TEAM", team)
            game_info.insert(1, "DATE", date)
            game_info.rename(columns={"G": "GAME"}, inplace=True)
            # convert opponent name to abbreviation
            opp_abbrev = TEAM_TO_TEAM_ABBR.get(str(game_info["Opponent"][i]).upper())
            # keep only certain columns
            game_info.insert(game_info.shape[1], "OPP", opp_abbrev)
            game_info = game_info[["TEAM", "GAME", "DATE", "HOME/AWAY", "OPP"]]
            game_info.insert(0, "GAME_ID", create_game_ID(game_info))
            game_info.set_index("GAME_ID", inplace=True)
    return game_info

# Function takes as input the team in question (abbrev.), the end-year of the season in question,
# and a list of the moving averages the user wants to use in the building of the model.
# For example, if the user wants to calculate the 3, 5, and 10 game moving averages for each collected
# stat, the user would enter [3,5,10] in the moving_avg_list function argument. Function will also
# calculate the cumulative average of each stat before each game by default.

# Get the game log (cleaned), both basic and advanced, for a single team in a single season
# function also has the option to add the next upcoming game to be predicted if the user
# sets add_upcoming to True. The date of the upcoming game defaults to today's date
# but can be ammended incase the you are predicting a the next game x days before (must
# still be after the most recent game has been played to avoid moving_avg errors)
def get_team_game_log(team, season_end_year, moving_avg_list, add_upcoming=False, 
                               date_upcoming=datetime.today().strftime('%Y-%m-%d')): 
    URL_basic = f'https://www.basketball-reference.com/teams/{team}/{season_end_year}/gamelog/'
    URL_advanced = f'https://www.basketball-reference.com/teams/{team}/{season_end_year}/gamelog-advanced/'

    page_basic = requests.get(URL_basic)
    page_advanced = requests.get(URL_advanced)

    soup_basic = BeautifulSoup(page_basic.content, 'html.parser')
    soup_advanced = BeautifulSoup(page_advanced.content, 'html.parser')

    gamelog_table_basic = soup_basic.find("table")
    gamelog_table_advanced = soup_advanced.find("table")

    # Adding proper column names to dataframe
    df_basic = pd.read_html(str(gamelog_table_basic))[0]
    df_basic.columns = ['RANK', 'GAME', 'DATE', 'HOME/AWAY', 'OPP', 'WIN/LOSS', 'TEAM_PTS', 'OPP_PTS', 'FGM', 
                        'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM', 'FTA', 'FT%', 'ORB', 'TRB', 'AST', 'STL', 
                        'BLK', 'TOV', 'PF', 'BLANK_COL', 'OPP_FGM', 'OPP_FGA', 'OPP_FG%', 'OPP_3PM', 'OPP_3PA',
                        'OPP_3P%', 'OPP_FTM', 'OPP_FTA', 'OPP_FT%', 'OPP_ORB', 'OPP_TRB', 'OPP_AST', 'OPP_STL', 
                        'OPP_BLK', 'OPP_TOV', 'OPP_PF']
    df_basic.insert(0, "TEAM", team)
    df_advanced = pd.read_html(str(gamelog_table_advanced))[0]
    df_advanced.columns = ['RANK', 'GAME', 'DATE', 'HOME/AWAY', 'OPP', 'WIN/LOSS', 'TEAM_PTS', 'OPP_PTS',
                           'O_RATING', 'D_RATING', 'PACE', 'FTA_RATE', '3PA_RATE', 'TS%', 'TRB%', 'AST%', 
                           'STL%', 'BLK%', 'BLANK', 'EFG%', 'TOV%', 'ORB%', 'FT/FGA', 'BLANK', 'OPP_EFG%', 
                           'OPP_TOV%', 'DRB%', 'OPP_FT/FGA']

    # Joining tables and cleaning up columns and rows
    df_final = df_basic.join(df_advanced, how='left', rsuffix = '_A')
    df_final.drop(["RANK", "BLANK_COL", "RANK_A", "GAME_A", "DATE_A", "HOME/AWAY_A", "OPP_A", "WIN/LOSS_A", 
                   "TEAM_PTS_A","OPP_PTS_A", "BLANK"], inplace=True, axis=1) 
    df_final["GAME"] = pd.to_numeric(df_final["GAME"], errors='coerce')
    df_final = df_final.dropna(subset=["GAME"])
    df_final["GAME"] = df_final["GAME"].astype(int)
    game_ID = create_game_ID(df_final)
    df_final.insert(0, "GAME_ID", game_ID)
    df_final.set_index("GAME_ID", inplace = True)
    
    if add_upcoming:
        upcoming_game_field = get_upcoming_game(team, date=date_upcoming)
        # only add the extra row if the team had a game on the specific date to avoid erros
        if not upcoming_game_field.empty:
            df_final = df_final.append(upcoming_game_field, sort=False)
    
    df_final = multiple_moving_averages(df_final, moving_avg_list)
      
    return df_final