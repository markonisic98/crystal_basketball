import pandas as pd
# Function for gameline betting naive (simple) strategy using random forest.
# Gives you the option to bet a certain percentage of your investment or certain amount per bet 
# (per game in naive betting strategy)
# It will use percentage if both of these variables are filled in by user, and investment will not
# change if neither of these values are filled in.
# there is also a withdraw_at_return paramater which will withdraw your excess investment return at
# a specific percentage return
# It is auto-set to 0.5 (50%). If you invest 1000, and your balance reaches 1500 or more at any point, 
# you will withdraw 500, and continue betting on the remainder of the games.
# The dataframe inputted into the function must have matching indices/length to the dataframe the model was tested on
def rf_gameline_simple_betting(win_loss_preds, df, investment=1000, bet_per_game_percentage = 0,
                         bet_per_game_amount = 0, withdraw_at_return = 0.5 , win_loss_column="HOME_TEAM_WIN/LOSS", 
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
            
        if WL == 'W' and WL_pred == 'W':
            if bet_per_game_percentage != 0:
                temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                + (starting_investment_balance[i] * bet_per_game_percentage * home_odds))
                ending_investment_balance[i] = temp_ending_balance
            else: # If the user elects to use bet amount instead of percentage
                temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                * (home_odds - 1)))
                ending_investment_balance[i] = temp_ending_balance
        elif WL == 'L' and WL_pred == 'L':
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

# Function for random forest gameline betting exploiting model odds vs market odds: an advanced strategy
# If the random forest model's implied odds of one team winning are sufficiently different (x%) from the
# market odds, then the model will bet on that team to win
def rf_gameline_advanced_betting(win_loss_preds, win_loss_prob_preds, df, investment=1000, 
                                 percentage_diff_to_exploit = 0.2, bet_per_game_percentage = 0,
                            bet_per_game_amount = 0, withdraw_at_return = 0.5, win_loss_column="HOME_TEAM_WIN/LOSS", 
                     home_odds_column="HOME_ODDS", away_odds_column="AWAY_ODDS", print_vals=False):
    
    # Making columns for your model's implied odds of each team winning based on model probability
    model_odds_of_win = []
    model_odds_of_loss = []
    for prob_loss, prob_win, home_odds, away_odds in zip(win_loss_prob_preds[:,0], win_loss_prob_preds[:,1], 
                                   df[home_odds_column], df[away_odds_column]):
        model_odds_of_loss.append(1/prob_loss)
        model_odds_of_win.append(1/prob_win)
    
    profit = 0 
    i = 0
    starting_investment_balance = [None]*len(win_loss_preds)
    starting_investment_balance[0] = investment
    ending_investment_balance = [None]*len(win_loss_preds)
    
    for WL, market_odds_H, market_odds_A, model_odds_H, model_odds_A in zip(df[win_loss_column], 
   df[home_odds_column], df[away_odds_column], model_odds_of_win, model_odds_of_loss):
        
        # Double check for odds error/outrageous and probably incorrect odds
        if (market_odds_H > 25 or market_odds_A > 25) or (market_odds_H < 1 or market_odds_A < 1):
            market_odds_H = 1.91
            market_odds_A = 1.91
        
        # If model suggests away team is X% more likely to win than market suggests, bet on away
        if model_odds_A*(1 + percentage_diff_to_exploit) < market_odds_A:
            if WL == 'L': #away wins
                if bet_per_game_percentage != 0: # If user uses bet percentage rather than amount strategy
                    temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                    + starting_investment_balance[i] * bet_per_game_percentage * market_odds_A)
                    ending_investment_balance[i] = temp_ending_balance
                else: 
                    temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                    * (market_odds_A - 1)))
                    ending_investment_balance[i] = temp_ending_balance
            else: # Home wins
                if bet_per_game_percentage !=0:
                    ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                    - bet_per_game_percentage))
                else: # If the user elects to use bet amount instead of percentage
                    ending_investment_balance[i] = (starting_investment_balance[i]
                    - bet_per_game_amount)
        # If model suggests home team is X% more likely to win than model suggests, bet on home
        elif model_odds_H*(1 + percentage_diff_to_exploit) < market_odds_H:
            if WL == 'W': # Home wins
                if bet_per_game_percentage != 0: # If user uses bet percentage rather than amount strategy
                    temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                    + starting_investment_balance[i] * bet_per_game_percentage * market_odds_H)
                    ending_investment_balance[i] = temp_ending_balance
                else: 
                    temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                    * (market_odds_H - 1)))
                    ending_investment_balance[i] = temp_ending_balance
            else: # Away wins
                if bet_per_game_percentage !=0:
                    ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                    - bet_per_game_percentage))
                else: # If the user elects to use bet amount instead of percentage
                    ending_investment_balance[i] = (starting_investment_balance[i]
                    - bet_per_game_amount)
        else: # No bet for now, later maybe change to bet the naive way
            ending_investment_balance[i] = starting_investment_balance[i]

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
    
# Naive betting strategy on Over/Under betting only based on model prediction of total points
# Function also supports simple betting strategy, where you assign a percentage difference
# in predictions of over under to be necessary to bet on the game
# for example, only bet on the over/under if model predicts total points 5%+ greater or smaller 
# than the over/under
# Whole function assumes betting site offers 1.91 odds on over/unders, as verified by bet365
def rf_over_under_simple_betting(total_pts_preds, df, investment=1000, bet_per_game_percentage = 0, 
                                percentage_diff_to_exploit = 0, bet_per_game_amount = 0, withdraw_at_return = 0.2,
                                 total_pts_column="TOTAL_PTS", 
                            over_under_column="OVER_UNDER", print_vals=False):
    
    profit = 0 
    i = 0
    starting_investment_balance = [None]*len(total_pts_preds)
    starting_investment_balance[0] = investment
    ending_investment_balance = [None]*len(total_pts_preds)
    for pred_total_pts, over_under, actual_total_pts in zip(total_pts_preds, df[over_under_column], df[total_pts_column]):
        if actual_total_pts == over_under or pred_total_pts == over_under:
            ending_investment_balance[i] = starting_investment_balance[i] # Push or don't bet
        else:
            if pred_total_pts > over_under*(1 + percentage_diff_to_exploit): 
                if actual_total_pts > over_under:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # You lose
                    if bet_per_game_percentage !=0:
                        ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                        - bet_per_game_percentage))
                    else: # If the user elects to use bet amount instead of percentage
                        ending_investment_balance[i] = (starting_investment_balance[i]
                        - bet_per_game_amount)
            elif pred_total_pts*(1 + percentage_diff_to_exploit) < over_under: 
                if actual_total_pts < over_under:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # You lose
                    if bet_per_game_percentage !=0:
                        ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                        - bet_per_game_percentage))
                    else: # If the user elects to use bet amount instead of percentage
                        ending_investment_balance[i] = (starting_investment_balance[i]
                        - bet_per_game_amount)   
            else:
                # You don't bet
                ending_investment_balance[i] = starting_investment_balance[i]
                    
        i += 1
        # Pull excess cash out when you are X% above your investment
        # If user wants to withdraw periodically to lock in return
        if withdraw_at_return != 0:
            if ending_investment_balance[i-1] > investment*(1 + withdraw_at_return) and i < len(total_pts_preds):
                profit += ending_investment_balance[i-1] - investment
                starting_investment_balance[i] = investment
            elif i < len(total_pts_preds):
                starting_investment_balance[i] = ending_investment_balance[i-1]
        else:
            if i < len(total_pts_preds):
                starting_investment_balance[i] = ending_investment_balance[i-1]
    
    profit += ending_investment_balance[i-1]            
    if print_vals:
        print(ending_investment_balance)
        print(profit)
    
    profit -= investment
    return ending_investment_balance, profit

# function for more advanced betting on over/under, using both the individual predictions of home 
# team points scored and away team points scored, combined with the prediction of total points 
# scored, with total_points_scored having to reach an optional threshold, defined by a percentage difference
# in market/bookie over/under and model over/under to actually bet
def rf_over_under_advanced_betting(total_pts_preds, home_pts_preds, away_pts_preds, df, investment=1000, 
                                   bet_per_game_percentage = 0, percentage_diff_to_exploit = 0, 
                                   bet_per_game_amount = 0, withdraw_at_return = 0.2,
                                 total_pts_column="TOTAL_PTS", 
                            over_under_column="OVER_UNDER", print_vals=False):
    
    profit = 0 
    i = 0
    starting_investment_balance = [None]*len(total_pts_preds)
    starting_investment_balance[0] = investment
    ending_investment_balance = [None]*len(total_pts_preds)
    for pred_total_pts, pred_home_pts, pred_away_pts, over_under, actual_total_pts in zip(total_pts_preds, 
                                                      home_pts_preds, away_pts_preds, df[over_under_column],
                                                                                df[total_pts_column]):
        if actual_total_pts == over_under or pred_total_pts == over_under:
            ending_investment_balance[i] = starting_investment_balance[i] # Push or don't bet
        else:
            if (pred_total_pts > over_under*(1 + percentage_diff_to_exploit) and 
                                                (pred_home_pts + pred_away_pts) > over_under): 
                if actual_total_pts > over_under:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # You lose
                    if bet_per_game_percentage !=0:
                        ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                        - bet_per_game_percentage))
                    else: # If the user elects to use bet amount instead of percentage
                        ending_investment_balance[i] = (starting_investment_balance[i]
                        - bet_per_game_amount)
            elif (pred_total_pts*(1 + percentage_diff_to_exploit) < over_under and 
                                                    (pred_home_pts + pred_away_pts) < over_under): 
                if actual_total_pts < over_under:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # You lose
                    if bet_per_game_percentage !=0:
                        ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                        - bet_per_game_percentage))
                    else: # If the user elects to use bet amount instead of percentage
                        ending_investment_balance[i] = (starting_investment_balance[i]
                        - bet_per_game_amount)   
            else:
                #you don't bet
                ending_investment_balance[i] = starting_investment_balance[i]
                    
        i += 1
        # Pull excess cash out when you are X% above your investment
        # If user wants to withdraw periodically to lock in return
        if withdraw_at_return != 0:
            if ending_investment_balance[i-1] > investment*(1 + withdraw_at_return) and i < len(total_pts_preds):
                profit += ending_investment_balance[i-1] - investment
                starting_investment_balance[i] = investment
            elif i < len(total_pts_preds):
                starting_investment_balance[i] = ending_investment_balance[i-1]
        else:
            if i < len(total_pts_preds):
                starting_investment_balance[i] = ending_investment_balance[i-1]
    
    profit += ending_investment_balance[i-1]            
    if print_vals:
        print(ending_investment_balance)
        print(profit)
    
    profit -= investment
    return ending_investment_balance, profit

# Function for naive/simple betting on game spread, using only the home spread prediction model
def rf_home_spread_simple_betting(home_spread_preds, df, investment=1000, bet_per_game_percentage = 0, 
                                percentage_diff_to_exploit = 0, bet_per_game_amount = 0, withdraw_at_return = 0.2,
                                 home_spread_column="HOME_TEAM_SPREAD", 
                            market_home_spread_column="IMPLIED_HOME_SPREAD", print_vals=False):
    
    profit = 0 
    i = 0
    starting_investment_balance = [None]*len(home_spread_preds)
    starting_investment_balance[0] = investment
    ending_investment_balance = [None]*len(home_spread_preds)
    for pred_home_spread, market_home_spread, actual_home_spread in zip(home_spread_preds, df[market_home_spread_column],
                                                                        df[home_spread_column]):
        # if one spread is positive and the other negative, in all cases it means that it passes the difference threshold
        if market_home_spread > 0:
            if pred_home_spread*(1 + percentage_diff_to_exploit) < market_home_spread:
                # betting against
                if actual_home_spread < market_home_spread:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # Push
                    if actual_home_spread == market_home_spread:
                        ending_investment_balance[i] = starting_investment_balance[i]
                    else:
                    # You lose
                        if bet_per_game_percentage !=0:
                            ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                            - bet_per_game_percentage))
                        else: # If the user elects to use bet amount instead of percentage
                            ending_investment_balance[i] = (starting_investment_balance[i]
                            - bet_per_game_amount)
            elif pred_home_spread > market_home_spread*(1 + percentage_diff_to_exploit):
                # betting to cover
                if actual_home_spread > market_home_spread:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # Push
                    if actual_home_spread == market_home_spread:
                        ending_investment_balance[i] = starting_investment_balance[i]
                    else:
                        # You lose
                        if bet_per_game_percentage !=0:
                            ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                            - bet_per_game_percentage))
                        else: # If the user elects to use bet amount instead of percentage
                            ending_investment_balance[i] = (starting_investment_balance[i]
                            - bet_per_game_amount)
            else:
                # You don't bet or it's a push
                ending_investment_balance[i] = starting_investment_balance[i]
        else:
            if pred_home_spread*(1 + percentage_diff_to_exploit) > market_home_spread:
                # betting against
                if actual_home_spread > market_home_spread:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # Push
                    if actual_home_spread == market_home_spread:
                        ending_investment_balance[i] = starting_investment_balance[i]
                    else:
                        # You lose
                        if bet_per_game_percentage !=0:
                            ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                            - bet_per_game_percentage))
                        else: # If the user elects to use bet amount instead of percentage
                            ending_investment_balance[i] = (starting_investment_balance[i]
                            - bet_per_game_amount)
            elif pred_home_spread < market_home_spread*(1 + percentage_diff_to_exploit):
                # betting to cover
                if actual_home_spread < market_home_spread:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # Push
                    if actual_home_spread == market_home_spread:
                        ending_investment_balance[i] = starting_investment_balance[i]
                    else:
                        # You lose
                        if bet_per_game_percentage !=0:
                            ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                            - bet_per_game_percentage))
                        else: # If the user elects to use bet amount instead of percentage
                            ending_investment_balance[i] = (starting_investment_balance[i]
                            - bet_per_game_amount)
            else:
                # You don't bet pr its a push 
                ending_investment_balance[i] = starting_investment_balance[i]
                    
        i += 1
        # Pull excess cash out when you are X% above your investment
        # If user wants to withdraw periodically to lock in return
        if withdraw_at_return != 0:
            if ending_investment_balance[i-1] > investment*(1 + withdraw_at_return) and i < len(home_spread_preds):
                profit += ending_investment_balance[i-1] - investment
                starting_investment_balance[i] = investment
            elif i < len(home_spread_preds):
                starting_investment_balance[i] = ending_investment_balance[i-1]
        else:
            if i < len(home_spread_preds):
                starting_investment_balance[i] = ending_investment_balance[i-1]
    
    profit += ending_investment_balance[i-1]            
    if print_vals:
        print(ending_investment_balance)
        print(profit)
    
    profit -= investment
    return ending_investment_balance, profit

# Function for more advanced betting on over/under, using both the individual predictions of home 
# team points scored and away team points scored, combined with the prediction of total points 
# scored, with total_points_scored having to reach an optional threshold, defined by a percentage difference
# in market/bookie over/under and model over/under to actually bet
def rf_home_spread_advanced_betting(home_spread_preds, home_pts_preds, away_pts_preds, df, investment=1000, 
                       bet_per_game_percentage = 0, percentage_diff_to_exploit = 0, bet_per_game_amount = 0, 
                         withdraw_at_return = 0.2, home_spread_column="HOME_TEAM_SPREAD", 
                            market_home_spread_column="IMPLIED_HOME_SPREAD", print_vals=False):
    
    profit = 0 
    i = 0
    starting_investment_balance = [None]*len(home_spread_preds)
    starting_investment_balance[0] = investment
    ending_investment_balance = [None]*len(home_spread_preds)
    for pred_home_spread, home_pts_preds, away_pts_preds, market_home_spread, actual_home_spread in zip(
                                home_spread_preds, home_pts_preds,
                               away_pts_preds, df[market_home_spread_column], df[home_spread_column]):
        # if one spread is positive and the other negative, in all cases it means that it passes the difference threshold
        if market_home_spread > 0:
            if (pred_home_spread*(1 + percentage_diff_to_exploit) < market_home_spread and 
                                                          (home_pts_preds - away_pts_preds) < market_home_spread):
                # betting against
                if actual_home_spread < market_home_spread:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # Push
                    if actual_home_spread == market_home_spread:
                        ending_investment_balance[i] = starting_investment_balance[i]
                    else:
                        # You lose
                        if bet_per_game_percentage !=0:
                            ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                            - bet_per_game_percentage))
                        else: # If the user elects to use bet amount instead of percentage
                            ending_investment_balance[i] = (starting_investment_balance[i]
                            - bet_per_game_amount)
            elif (pred_home_spread > market_home_spread*(1 + percentage_diff_to_exploit) and 
                                                    (home_pts_preds - away_pts_preds) > market_home_spread):
                # betting to cover
                if actual_home_spread > market_home_spread:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # Push
                    if actual_home_spread == market_home_spread:
                        ending_investment_balance[i] = starting_investment_balance[i]
                    else:
                        # You lose
                        if bet_per_game_percentage !=0:
                            ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                            - bet_per_game_percentage))
                        else: # If the user elects to use bet amount instead of percentage
                            ending_investment_balance[i] = (starting_investment_balance[i]
                            - bet_per_game_amount)
            else:
                # You don't bet or it's a push
                ending_investment_balance[i] = starting_investment_balance[i]
        else:
            if (pred_home_spread*(1 + percentage_diff_to_exploit) > market_home_spread and 
                                                          (home_pts_preds - away_pts_preds) > market_home_spread):
                # betting against
                if actual_home_spread > market_home_spread:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # Push
                    if actual_home_spread == market_home_spread:
                        ending_investment_balance[i] = starting_investment_balance[i]
                    else:
                        # You lose
                        if bet_per_game_percentage !=0:
                            ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                            - bet_per_game_percentage))
                        else: # If the user elects to use bet amount instead of percentage
                            ending_investment_balance[i] = (starting_investment_balance[i]
                            - bet_per_game_amount)
            elif (pred_home_spread < market_home_spread*(1 + percentage_diff_to_exploit) and 
                                                    (home_pts_preds - away_pts_preds) < market_home_spread):
                # betting to cover
                if actual_home_spread < market_home_spread:
                    # You win
                    if bet_per_game_percentage != 0: # If user users bet percentage rather than amount strategy
                        temp_ending_balance = (starting_investment_balance[i] * (1 - bet_per_game_percentage)
                        + starting_investment_balance[i] * bet_per_game_percentage * 1.91)
                        ending_investment_balance[i] = temp_ending_balance
                    else: # If user uses bet amount rather than bet percentage
                        temp_ending_balance = (starting_investment_balance[i] + (bet_per_game_amount 
                        * (1.91 - 1)))
                        ending_investment_balance[i] = temp_ending_balance
                else:
                    # Push
                    if actual_home_spread == market_home_spread:
                        ending_investment_balance[i] = starting_investment_balance[i]
                    else:
                        # You lose
                        if bet_per_game_percentage !=0:
                            ending_investment_balance[i] = (starting_investment_balance[i] * (1 
                            - bet_per_game_percentage))
                        else: # If the user elects to use bet amount instead of percentage
                            ending_investment_balance[i] = (starting_investment_balance[i]
                            - bet_per_game_amount)
            else:
                # You don't bet pr its a push 
                ending_investment_balance[i] = starting_investment_balance[i]
        i += 1
        # Pull excess cash out when you are X% above your investment
        # If user wants to withdraw periodically to lock in return
        if withdraw_at_return != 0:
            if ending_investment_balance[i-1] > investment*(1 + withdraw_at_return) and i < len(home_spread_preds):
                profit += ending_investment_balance[i-1] - investment
                starting_investment_balance[i] = investment
            elif i < len(home_spread_preds):
                starting_investment_balance[i] = ending_investment_balance[i-1]
        else:
            if i < len(home_spread_preds):
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

