# benchmarks to compare prediction accuracy to.
# Prediction accuracy is based on scraped market odds, spreads, and over/unders vs. actual result
from sklearn import metrics
import numpy as np
# accuracy of market predictions, based on odds favorite winning or losing
def market_winner_accuracy(df, home_odds_column_name="HOME_ODDS", away_odds_column_name="AWAY_ODDS",
                          win_loss_column_name="HOME_TEAM_WIN/LOSS"):
    # number of games market predicted 50/50 chance of winning (even odds)
    even_matchups = df[df[home_odds_column_name] == df[away_odds_column_name]]
    num_even_predictions = even_matchups.shape[0]

    # percentage correct home game predictions by market
    home_game_wins = df[df[win_loss_column_name] == 'W']
    home_game_wins_correct_preds = df[(df[win_loss_column_name] == 'W') & (df[home_odds_column_name] < df[away_odds_column_name])]
    home_win_pred_accuracy = home_game_wins_correct_preds.shape[0]/home_game_wins.shape[0]
    print("Home win accuracy: ", home_win_pred_accuracy)

    # percentage correct away game predictions by market
    away_game_wins = df[df[win_loss_column_name] == 'L']
    away_game_wins_correct_preds = df[(df[win_loss_column_name] == 'L') & (df[home_odds_column_name] > df[away_odds_column_name])]
    away_win_pred_accuracy = away_game_wins_correct_preds.shape[0]/away_game_wins.shape[0]
    print("Away win accuracy: ", away_win_pred_accuracy)

    # overall win (moneyline/gameline) prediction accuracy
    winner_correct_preds = df[((df[win_loss_column_name] == 'W') & (df[home_odds_column_name] < df[away_odds_column_name]))
                                  | ((df[win_loss_column_name] == 'L') & (df[home_odds_column_name] > df[away_odds_column_name]))]
    market_winner_prediction_accuracy = winner_correct_preds.shape[0]/(df.shape[0] - num_even_predictions)
    print("Market win prediction accuracy: ", market_winner_prediction_accuracy)

# measuring market over/under prediction accuracy
# this is going to be the benchmark for our over/under (total points) predictions
def market_over_under_accuracy(df, total_pts_column_name="TOTAL_PTS", over_under_column_name="OVER_UNDER"):
    
    under_games = df[df[total_pts_column_name] < df[over_under_column_name]]
    over_games = df[df[total_pts_column_name] > df[over_under_column_name]]
    push_games = df[df[total_pts_column_name] == df[over_under_column_name]]

    under_percentage = under_games.shape[0]/df.shape[0]
    print("Under %: ", under_percentage*100, "%")
    over_percentage = over_games.shape[0]/df.shape[0]
    print("Over %: ", over_percentage*100, "%")
    push_percentage = push_games.shape[0]/df.shape[0]
    print("Push %: ", push_percentage*100, "%")

    # mean total points market errors to be used as a benchmark
    mean_absolute_over_under_error = metrics.mean_absolute_error(df[total_pts_column_name], df[over_under_column_name])
    print("Mean abs. error: ", mean_absolute_over_under_error)
    mean_squared_over_under_error = metrics.mean_squared_error(df[total_pts_column_name], df[over_under_column_name])
    print("Mean squared error: ", mean_squared_over_under_error)
    root_mean_squared_over_under_error = np.sqrt(metrics.mean_squared_error(df[total_pts_column_name], df[over_under_column_name]))
    print("Root mean squared error: ", root_mean_squared_over_under_error)

# find the error rate from the points spread predictions
# this is going to be the benchmark for our home points spread predictions
def market_spread_accuracy(df, actual_home_spread_column_name="HOME_TEAM_SPREAD", market_home_spread_column_name="IMPLIED_HOME_SPREAD"):
    cover_games = df[df[actual_home_spread_column_name] < df[market_home_spread_column_name]]
    against_games = df[df[actual_home_spread_column_name] > df[market_home_spread_column_name]]
    push_games = df[df[actual_home_spread_column_name] == df[market_home_spread_column_name]]

    cover_percentage = cover_games.shape[0]/df.shape[0]
    print("Cover %: ", cover_percentage*100, "%")
    against_percentage = against_games.shape[0]/df.shape[0]
    print("Against %: ", against_percentage*100, "%")
    push_percentage = push_games.shape[0]/df.shape[0]
    print("Push %: ", push_percentage*100, "%")

    # mean total points market errors to be used as a benchmark
    mean_absolute_spread_error = metrics.mean_absolute_error(df[actual_home_spread_column_name], df[market_home_spread_column_name])
    print("Mean abs. error: ", mean_absolute_spread_error)
    mean_squared_spread_error = metrics.mean_squared_error(df[actual_home_spread_column_name], df[market_home_spread_column_name])
    print("Mean squared error: ", mean_squared_spread_error)
    root_mean_squared_spread_error = np.sqrt(metrics.mean_squared_error(df[actual_home_spread_column_name], df[market_home_spread_column_name]))
    print("Root mean squared error: ", root_mean_squared_spread_error)

# this function will call upon a headless browser, make sure you have selenium and chromedriver installed to use this
# only accepts season_end_years {2018,...,2021}. Only seasons fivethirtyeight made these 
# kinds of predictions
def fivethirtyeight_winner_and_spread_accuracy(fivethirtyeight_scraper_results):
    
    match_date_list = fivethirtyeight_scraper_results[0]
    advanced_pred_name = fivethirtyeight_scraper_results[1]
    basic_pred_name = fivethirtyeight_scraper_results[2]
    advanced_home_win_prob = fivethirtyeight_scraper_results[3]
    advanced_away_win_prob = fivethirtyeight_scraper_results[4]
    advanced_home_spread_pred = fivethirtyeight_scraper_results[5]
    basic_home_win_prob = fivethirtyeight_scraper_results[6]
    basic_away_win_prob = fivethirtyeight_scraper_results[7]
    basic_home_spread_pred = fivethirtyeight_scraper_results[8]
    match_home_team_list = fivethirtyeight_scraper_results[9]
    match_away_team_list = fivethirtyeight_scraper_results[10]
    match_home_team_pts_list = fivethirtyeight_scraper_results[11]
    match_away_team_pts_list = fivethirtyeight_scraper_results[12]
    
    if basic_pred_name == "EMPTY": # only one prediction method (advanced: CARMELO)
        home_correct_guesses = 0
        away_correct_guesses = 0
        equal_prob = 0
        #how well do they predict the spread
        cover_spread = 0
        against_spread = 0
        equal_spread = 0
        for i in range(len(match_date_list)):
            if advanced_home_win_prob[i] > advanced_away_win_prob[i] and match_home_team_pts_list[i] > match_away_team_pts_list[i]:
                home_correct_guesses += 1
            elif advanced_home_win_prob[i] < advanced_away_win_prob[i] and match_home_team_pts_list[i] < match_away_team_pts_list[i]:
                away_correct_guesses += 1
            elif advanced_home_win_prob[i] == advanced_away_win_prob[i]:
                equal_prob += 1
            
            if advanced_home_spread_pred[i] < (match_away_team_pts_list[i] - match_home_team_pts_list[i]):
                cover_spread += 1
            elif advanced_home_spread_pred[i] > (match_away_team_pts_list[i] - match_home_team_pts_list[i]):
                against_spread += 1
            else: # push
                equal_spread += 1
                
        advanced_accuracy = (home_correct_guesses + away_correct_guesses)/(len(match_date_list)-equal_prob)
        print(advanced_pred_name, " winner accuracy: ", advanced_accuracy)
        cover_percentage = cover_spread/(len(match_date_list))
        against_percentage = against_spread/(len(match_date_list))
        equal_percentage = equal_spread/(len(match_date_list))
        print(advanced_pred_name, " Cover %: ", cover_percentage)
        print(advanced_pred_name, " Against %: ", against_percentage)
        print(advanced_pred_name, " Equal Spread %: ", equal_percentage)
                                             
    else: # for any other season, two predictors
        home_correct_guesses_advanced = 0
        away_correct_guesses_advanced = 0
        equal_prob_advanced = 0
        home_correct_guesses_basic = 0
        away_correct_guesses_basic = 0
        equal_prob_basic = 0
        #how well do they predict the spread
        cover_spread_advanced = 0
        against_spread_advanced = 0
        equal_spread_advanced = 0
        cover_spread_basic = 0
        against_spread_basic = 0
        equal_spread_basic = 0
        for i in range(len(match_date_list)):
            if advanced_home_win_prob[i] > advanced_away_win_prob[i] and match_home_team_pts_list[i] > match_away_team_pts_list[i]:
                home_correct_guesses_advanced += 1
            elif advanced_home_win_prob[i] < advanced_away_win_prob[i] and match_home_team_pts_list[i] < match_away_team_pts_list[i]:
                away_correct_guesses_advanced += 1
            elif advanced_home_win_prob[i] == advanced_away_win_prob[i]:
                equal_prob_advanced += 1
                
            if basic_home_win_prob[i] > basic_away_win_prob[i] and match_home_team_pts_list[i] > match_away_team_pts_list[i]:
                home_correct_guesses_basic += 1
            elif basic_home_win_prob[i] < basic_away_win_prob[i] and match_home_team_pts_list[i] < match_away_team_pts_list[i]:
                away_correct_guesses_basic += 1
            elif basic_home_win_prob[i] == basic_away_win_prob[i]:
                equal_prob_basic += 1
                
            if advanced_home_spread_pred[i] < (match_away_team_pts_list[i] - match_home_team_pts_list[i]):
                cover_spread_advanced += 1
            elif advanced_home_spread_pred[i] > (match_away_team_pts_list[i] - match_home_team_pts_list[i]):
                against_spread_advanced += 1
            else: # push
                equal_spread_advanced += 1
                
            if basic_home_spread_pred[i] < (match_away_team_pts_list[i] - match_home_team_pts_list[i]):
                cover_spread_basic += 1
            elif basic_home_spread_pred[i] > (match_away_team_pts_list[i] - match_home_team_pts_list[i]):
                against_spread_basic += 1
            else: # push
                equal_spread_basic += 1
        advanced_accuracy = (home_correct_guesses_advanced + away_correct_guesses_advanced)/(len(match_date_list)-equal_prob_advanced)
        basic_accuracy = (home_correct_guesses_basic + away_correct_guesses_basic)/(len(match_date_list)-equal_prob_basic)
        cover_percentage_advanced = cover_spread_advanced/(len(match_date_list))
        against_percentage_advanced = against_spread_advanced/(len(match_date_list))
        equal_percentage_advanced = equal_spread_advanced/(len(match_date_list))
        cover_percentage_basic = cover_spread_basic/(len(match_date_list))
        against_percentage_basic = against_spread_basic/(len(match_date_list))
        equal_percentage_basic = equal_spread_basic/(len(match_date_list))
        print(advanced_pred_name, " Cover %: ", cover_percentage_advanced)
        print(advanced_pred_name, " Against %: ", against_percentage_advanced)
        print(advanced_pred_name, " Equal Spread %: ", equal_percentage_advanced)
        
        print(basic_pred_name, " Cover %: ", cover_percentage_basic)
        print(basic_pred_name, " Against %: ", against_percentage_basic)
        print(basic_pred_name, " Equal Spread %: ", equal_percentage_basic)
        
        print(advanced_pred_name, " winner accuracy: ", advanced_accuracy)
        
        print(basic_pred_name, " winner accuracy: ", basic_accuracy)