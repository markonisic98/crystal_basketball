import pandas as pd
# Function for simple gameline betting using the SVM classification predictions
def svm_gameline_simple_betting(win_loss_preds, df, investment=1000, bet_per_game_percentage = 0,
                        bet_per_game_amount = 0, withdraw_at_return = 0.5, win_loss_column="HOME_TEAM_WIN/LOSS", 
                     home_odds_column="HOME_ODDS", away_odds_column="AWAY_ODDS", print_vals=False):
    
    profit = 0 
    i = 0
    starting_investment_balance = [None]*len(win_loss_preds)
    starting_investment_balance[0] = investment
    ending_investment_balance = [None]*len(win_loss_preds)
    for WL, WL_pred, home_odds, away_odds in zip(df[win_loss_column], win_loss_preds, 
                                                  df[home_odds_column], df[away_odds_column]):
        
        # Double check for odds error/outrageous and probably incorrect odds
        if (home_odds > 25 or away_odds > 25) or (home_odds < 1 or away_odds < 1):
            home_odds = 1.91
            away_odds = 1.91
            
        if WL == 'W' and WL_pred == 1:
            if bet_per_game_percentage != 0:
                temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                + (starting_investment_balance[i] * bet_per_game_percentage * home_odds))
                ending_investment_balance[i] = temp_ending_balance
            else: # If the user elects to use bet amount instead of percentage
                temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                * (home_odds - 1)))
                ending_investment_balance[i] = temp_ending_balance
        elif WL == 'L' and WL_pred == 0:
            if bet_per_game_percentage !=0:
                temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                + starting_investment_balance[i] * bet_per_game_percentage * away_odds)
                ending_investment_balance[i] = temp_ending_balance
            else: # If the user elects to use bet amount instead of percentage
                temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                * (away_odds - 1)))
                ending_investment_balance[i] = temp_ending_balance
        else:
            if bet_per_game_percentage !=0:
                ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                - bet_per_game_percentage))
            else: # If the user elects to use bet amount instead of percentage
                ending_investment_balance[i] = (starting_investment_balance[i]
                - bet_per_game_amount)
        
        i += 1
        # Pull excess cash out when you are X% above your investment
        # If user wants to withdraw periodically to lock in return
        if withdraw_at_return != 0:
            if ending_investment_balance[i-1] > investment*(1 + withdraw_at_return) and i < len(win_loss_preds):
                profit += ending_investment_balance[i-1] - investment
                starting_investment_balance[i] = investment
            elif i < len(win_loss_preds):
                starting_investment_balance[i] = ending_investment_balance[i-1]
        else:
            if i < len(win_loss_preds):
                starting_investment_balance[i] = ending_investment_balance[i-1]
    
    profit += ending_investment_balance[i-1]            
    if print_vals:
        print(ending_investment_balance)
        print(profit)
    
    profit -= investment
    return ending_investment_balance, profit

# This function takes as input a dictionary of betting strategies, with the keys
# being the user's desired column name, and their respective values representing 
# the value of the investment for a given betting strategy over time, in a list
# If a user were to enter a random forest simple betting as one item in the dictionary,
# for example, then the user would name it something like: "RF_SIMPLE_BETTING_STRAT"
# This function assumes that the dataframe being passd into it has all of the relevant 
# information used to build the predictive models in the first place (i.e.: the dataset
# used to build the model, but with only the indices used in the test set for betting)
# The purpose of this function is to consolidate info for data analysis for the user
# for each strategy, but the dictionary can include any columns and relevant information
# For example, one column to include could be the model probability of a classification
# outcome, or the implied model odds of a certain result. This may be relevant for 
# data visualisation/analysis purposes
def create_betting_df(betting_strategies_dict, df):
    
    final_df = df[["GAME_ID"]]
    final_df.insert(final_df.shape[1], "HOME_TEAM", df["HOME_TEAM"])
    final_df.insert(final_df.shape[1], "AWAY_TEAM", df["AWAY_TEAM"])
    final_df.insert(final_df.shape[1], "HOME_TEAM_WIN/LOSS", df["HOME_TEAM_WIN/LOSS"])
    final_df.insert(final_df.shape[1], "HOME_TEAM_PTS", df["HOME_TEAM_PTS"])
    final_df.insert(final_df.shape[1], "AWAY_TEAM_PTS", df["AWAY_TEAM_PTS"])
    final_df.insert(final_df.shape[1], "TOTAL_PTS", df["TOTAL_PTS"])
    final_df.insert(final_df.shape[1], "HOME_TEAM_SPREAD", df["HOME_TEAM_SPREAD"])
    final_df.insert(final_df.shape[1], "HOME_ODDS_MARKET", df["HOME_ODDS"])
    final_df.insert(final_df.shape[1], "AWAY_ODDS_MARKET", df["AWAY_ODDS"])
    final_df.insert(final_df.shape[1], "OVER_UNDER_MARKET", df["OVER_UNDER"])
    final_df.insert(final_df.shape[1], "HOME_TEAM_SPREAD_MARKET", df["IMPLIED_HOME_SPREAD"])
    for column_name, betting_balance, in betting_strategies_dict.items():
        final_df.insert(final_df.shape[1], column_name, betting_balance)
    
    final_df.set_index("GAME_ID", inplace=True)
    #final_df.sort_index(axis=0, inplace=True)
    return final_df

