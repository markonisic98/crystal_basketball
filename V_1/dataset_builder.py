import pandas as pd
from team_abbreviations import TEAM_TO_TEAM_ABBR 
from scrapers import get_game_odds, get_team_game_log
from dataset_builder_helpers import (create_game_ID, cum_win_loss, win_percentage_moving_avg,
                                    gen_moving_avg, multiple_moving_averages, calc_matchup_record,
                                    calc_combined_pts_and_decimal_odds)

# Function to merge single game log with odds for that game
# Takes a team (abbrev), season-end-year, and print_progress input
# Print progress defaults to true - it will print your progress
# when building the final dataset
def get_game_log_and_odds(team, season_end_year, moving_avgs_list, print_progress=True):
    df = get_team_game_log(team, season_end_year, moving_avgs_list)
    team_implied_pts = []
    team_odds = []
    over_under = []
    team_spread = []
    opp_implied_pts = []
    opp_odds = []
    for i, each in enumerate(df['TEAM']):
        temp_list = get_game_odds(df['TEAM'][i], df['DATE'][i]) 
        team_implied_pts.append(temp_list[0])
        team_odds.append(temp_list[1])
        over_under.append(temp_list[2])
        team_spread.append(temp_list[3])
        opp_implied_pts.append(temp_list[4])
        opp_odds.append(temp_list[5])
    
    df.insert(8, "TEAM_IMPLIED_PTS", team_implied_pts)
    df.insert(9, "TEAM_ODDS", team_odds)
    df.insert(10, "OVER_UNDER", over_under)
    df.insert(11, "TEAM_SPREAD", team_spread)
    df.insert(12, "OPP_IMPLIED_PTS", opp_implied_pts)
    df.insert(13, "OPP_ODDS", opp_odds)
    
    # Print statements to show where the program is currently at
    if print_progress:
        print(f'{team}, {season_end_year} season loaded successfully')
    
    return df

# Function to make the total league dataset
# First splits the dataset of a single team by home games and away games, then connects all the 
# datapoints of each team's home games with its respective opposing (away) team
# Merging is done using the Game ID created for a game (e.g.: 2021-01-11DALUTA)
# Function takes as input a season_end_year, to collect a whole season's worth of data,
# and two optional integers first_game_to_collect, and last_game_to_collect, which will only keep
# games in the dataset in between these two numbers. This may provide more accurate predictions due to
# small sample size for averages in the beginning of the season, and rest/tanking/lost motivation
# near the end of the season.
def create_final_dataset(season_end_year, moving_avgs_list, first_game_to_collect=1, last_game_to_collect=82):
    final_dataset = pd.DataFrame()
    home_final_dataset = pd.DataFrame()
    away_final_dataset = pd.DataFrame()
    for team in TEAM_TO_TEAM_ABBR.values():
        team_dataset = get_game_log_and_odds(team, season_end_year, moving_avgs_list)
        home_team_dataset = team_dataset[team_dataset['HOME/AWAY'] != '@']
        away_team_dataset = team_dataset[team_dataset['HOME/AWAY'] == '@']
        home_final_dataset = home_final_dataset.append(home_team_dataset) 
        away_final_dataset = away_final_dataset.append(away_team_dataset) 
    final_dataset = home_final_dataset.join(away_final_dataset, how='left', rsuffix = '_OPP')
    final_dataset.sort_index(axis=0, inplace=True)
    final_dataset.insert(final_dataset.columns.get_loc("OPP_ODDS")+1, "PREVIOUS_MATCHUP_RECORD", 
                                                                 calc_matchup_record(final_dataset))
    cols_to_delete1 = final_dataset.columns.get_loc("FGM")
    cols_to_delete2 = final_dataset.columns.get_loc("OPP_FT/FGA")
    final_dataset = final_dataset.drop(final_dataset.iloc[:, cols_to_delete1:cols_to_delete2+1], axis = 1)  
    cols_to_delete3 = final_dataset.columns.get_loc("TEAM_OPP")
    cols_to_delete4 = final_dataset.columns.get_loc("OPP_FT/FGA_OPP")
    final_dataset = final_dataset.drop(final_dataset.iloc[:, cols_to_delete3:cols_to_delete4+1], axis = 1) 
    final_dataset = final_dataset.drop(["HOME/AWAY"], axis = 1)
    final_dataset = final_dataset.rename(columns={"TEAM" : "HOME_TEAM", "OPP" : "AWAY_TEAM", 
                                            "WIN/LOSS" : "HOME_TEAM_WIN/LOSS", "TEAM_PTS" : "HOME_TEAM_PTS",
                                                "OPP_PTS" : "AWAY_TEAM_PTS"})
    final_dataset['HOME_TEAM_PTS'] = pd.to_numeric(final_dataset['HOME_TEAM_PTS'])
    final_dataset['AWAY_TEAM_PTS'] = pd.to_numeric(final_dataset['AWAY_TEAM_PTS'])
    total_pts, home_spread, home_odds, away_odds = calc_combined_pts_and_decimal_odds(final_dataset)
    final_dataset.insert(final_dataset.columns.get_loc("TEAM_IMPLIED_PTS"), "TOTAL_PTS", total_pts)
    final_dataset.insert(final_dataset.columns.get_loc("TEAM_IMPLIED_PTS"), "HOME_TEAM_SPREAD", home_spread)
    final_dataset.insert(final_dataset.columns.get_loc("TEAM_IMPLIED_PTS"), "HOME_ODDS", home_odds)
    final_dataset.insert(final_dataset.columns.get_loc("TEAM_IMPLIED_PTS"), "AWAY_ODDS", away_odds)
    final_dataset = final_dataset.rename(columns={"TEAM_ODDS" : "HOME_ODDS_AMERICAN", 
                                        "OPP_ODDS" : "AWAY_ODDS_AMERICAN", "TEAM_IMPLIED_PTS": "HOME_IMPLIED_PTS",
                                    "OPP_IMPLIED_PTS": "AWAY_IMPLIED_PTS", "TEAM_SPREAD": "IMPLIED_HOME_SPREAD"})
    final_dataset = final_dataset[final_dataset['GAME'] >= first_game_to_collect] 
    final_dataset = final_dataset[final_dataset['GAME'] <= last_game_to_collect] 
    return final_dataset

# Final dataset for multiple seasons to get more training data
# Season_end_year_start is the first season end year to collect data on and season_end_year_end is last year (inclusive)
def multi_season_final_dataset(season_end_year_start, season_end_year_end, moving_avgs_list, 
                               first_game_to_collect=1, last_game_to_collect=82):
    final_df = pd.DataFrame()
    for i in range(season_end_year_start, season_end_year_end + 1):
        temp_df = create_final_dataset(i, moving_avgs_list, first_game_to_collect, last_game_to_collect)
        final_df = final_df.append(temp_df)
    return final_df
# Call df_with_odds = multi_season_final_dataset(2016,2020) as an example, to get V1 dataset
# to use in the building of a predictive model
# Run time is roughly 4 hours on a mac-air 2018

