from dataCollection.builder import multi_season_final_dataset

# Create dataset with all regular season games from the 2015/2016 season through
# to the end of the 2019/2020 season.
# This will take roughly 45 mins per season to import - by far the most computationally intensive function
# in the package (due to the scraping from various websites)
dataset_V1 = multi_season_final_dataset(2016, 2020, [5,10], 1, 82)

# Place the dataset in the folder for quicker pulling of data in future
dataset_V1.to_csv("dataset_V1.csv")

# Load the dataset from the folder
import pandas as pd
dataset_V1 = pd.read_csv("dataset_V1.csv")

# checking prediction benchmarks to compare to 
# fivethirtyeight predictions
from backtesting import benchmarks
from dataCollection import scrapers

# enter the filepath to your Google Chrome app, and your chromedriver path, respectively
chrome_app_filepath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
binary_filepath = "/Users/markonisic/opt/WebDriver/bin/chromedriver"

# Fivethirtyeight predictions include playoffs, whereas scraped market predictions don't
#----2017/2018 - 2020/2021 SEASON-----
season_end_years = [2018, 2019, 2020, 2021] # all the available fivethirtyeight season predictions
for year in season_end_years:
    print(f"-----{year-1}/{year} SEASON------")
    scraper_results = scrapers.fivethirtyeight_game_scraper(binary_filepath, chrome_app_filepath, year)
    benchmarks.fivethirtyeight_winner_and_spread_accuracy(scraper_results)
    
# market predictions 
from backtesting import benchmarks
from dataCollection import scrapers
#----OVERALL DATASET PREDICTIONS-----
print("----2015/2016 through 2019/2020 SEASON----")
benchmarks.market_winner_accuracy(dataset_V1)
benchmarks.market_over_under_accuracy(dataset_V1)
benchmarks.market_spread_accuracy(dataset_V1)

#-----YEARLY PREDICTIONS--------
season_end_years = [2016, 2017, 2018, 2019, 2020]
for year in season_end_years: # 5 seasons in this dataset
    print(f"-----{year-1}/{year} SEASON------") # COVID Season different schedule
    temp_set = dataset_V1[(dataset_V1["DATE"] < f"{year}-10-01") & 
                          (dataset_V1["DATE"] > f"{year-1}-10-01")]
    benchmarks.market_winner_accuracy(temp_set)
    benchmarks.market_over_under_accuracy(temp_set)
    benchmarks.market_spread_accuracy(temp_set)

# Load the dataset from the folder in a future session
from dataCollection.builder import create_final_dataset
from datetime import datetime
from calendar import month_abbr
# Using only games after the 10th of the season, to let rolling averages become useful
dataset_V1 = dataset_V1[dataset_V1['GAME'] >= 10]
# Predicting past games from this season
this_season = create_final_dataset(2021, [5,10], first_game_to_collect=10, add_upcoming=True)
todays_date = datetime.today().strftime('%Y-%m-%d')
this_season_past_games = this_season[this_season['DATE'] != todays_date]
# alternatively, to get past games from this season (not yet to occur), we could just
# use the same function and parameters as "this_season" above, but set "add_upcoming"
# parameter to False

# import functions to create models with the data
from backtesting.predictors.RF import (rf_classify_win_loss, feature_importance_sorted,
                                      rf_classify_win_loss_KFold, rf_predict_home_team_points,
                                      rf_predict_away_team_points, rf_predict_total_points,
                                      rf_predict_home_spread)


# Set a random state for reproducible results through various models
# so they can be combined into one dataframe later, predicting the same games
random_state = 1402

# Predict game outcome using random forest classifier
# 0.0075 min leaf samples seems to be optimal
rfc_results = rf_classify_win_loss(dataset_V1, random_state=random_state, min_samples_leaf=0.0075)
rf_game_predictions = rfc_results[0]
rf_game_prediction_probabilities = rfc_results[1]
rfc_model = rfc_results[2]
rfc_test_set = rfc_results[3]

# Test the accuracy of this model using repeated KFold testing with most of its
# parameters held to their default values
KFold_testing = rf_classify_win_loss_KFold(rfc_model, dataset_V1, random_state=random_state)
model_accuracy, model_std = KFold_testing[0], KFold_testing[1]
model_accuracy, model_std

from backtesting.predictors.SVM import svm_classify_win_loss, svm_classify_win_loss_KFold

# Building SVM winner classification model. C = approx. 10 seems optimal, with gamma='scale'
# Grid search was performed, but with very large intervals
svm_results = svm_classify_win_loss(dataset_V1, C=10, random_state=random_state)
svm_game_predictions = svm_results[0]
svm_model = svm_results[1]
svm_test_set = svm_results[2]

# Test the accuracy of this model using repeated KFold testing with most of its
# parameters held to their default values
KFold_testing = svm_classify_win_loss_KFold(svm_model, dataset_V1, random_state=random_state)
model_accuracy, model_std = KFold_testing[0], KFold_testing[1]
model_accuracy, model_std

# Pulling the game lineup for a specific team on a specific night
from dataCollection.scrapers import get_game_lineup

get_game_lineup('BRK', '2021-02-02')

# Making predictions/bets on the games played this season
# to keep track of bets/profit, if user has been making bets,
# or to test how well the user would have done so far betting this season
from backtesting.predictors.RF import rf_predict_total_points, rf_predict_home_spread
from backtesting.betting.RF import (create_betting_df, rf_gameline_advanced_betting, rf_gameline_simple_betting, rf_home_spread_advanced_betting,
                        rf_home_spread_simple_betting, rf_over_under_advanced_betting, rf_over_under_simple_betting)
from backtesting.betting.SVM import svm_gameline_simple_betting

X_features_this_season=this_season_past_games.iloc[:, this_season_past_games.columns.get_loc("PREVIOUS_MATCHUP_RECORD"):this_season_past_games.shape[1]]
y_this_season=this_season_past_games.iloc[:,this_season_past_games.columns.get_loc("HOME_TEAM_WIN/LOSS")]
# utilizing the predictive models
rf_gameline_preds_model = rf_classify_win_loss(dataset_V1, random_state=random_state, min_samples_leaf=0.0075)[2]
rf_gameline_preds = rf_gameline_preds_model.predict(X_features_this_season)
rf_gameline_prob_preds = rf_gameline_preds_model.predict_proba(X_features_this_season) #remember this is a 2d array/list
rf_home_pts_preds_model = rf_predict_home_team_points(dataset_V1, random_state=random_state)[1]
rf_home_pts_preds = rf_home_pts_preds_model.predict(X_features_this_season)
rf_away_pts_preds_model = rf_predict_away_team_points(dataset_V1, random_state=random_state)[1]
rf_away_pts_preds = rf_away_pts_preds_model.predict(X_features_this_season)
rf_spread_preds_model = rf_predict_home_spread(dataset_V1, random_state=random_state)[1]
rf_spread_preds = rf_spread_preds_model.predict(X_features_this_season)
rf_total_pts_preds_model = rf_predict_total_points(dataset_V1, random_state=random_state)[1]
rf_total_pts_preds = rf_total_pts_preds_model.predict(X_features_this_season)
svm_gameline_preds_model = svm_classify_win_loss(dataset_V1, C=10, random_state=random_state)[1]
svm_gameline_preds = svm_gameline_preds_model.predict(X_features_this_season)

# inputting predictions into betting strategy
rf_simple_gameline = rf_gameline_simple_betting(rf_gameline_preds, this_season_past_games, bet_per_game_percentage=0.1)
rf_advanced_gameline = rf_gameline_advanced_betting(rf_gameline_preds, rf_gameline_prob_preds, this_season_past_games, bet_per_game_percentage=0.1)
rf_simple_spread = rf_home_spread_simple_betting(rf_spread_preds, this_season_past_games, bet_per_game_percentage=0.1)
rf_advanced_spread = rf_home_spread_advanced_betting(rf_spread_preds, rf_home_pts_preds, rf_away_pts_preds, this_season_past_games, bet_per_game_percentage=0.1, percentage_diff_to_exploit=0.2)
rf_simple_over_under = rf_over_under_simple_betting(rf_total_pts_preds, this_season_past_games, bet_per_game_percentage=0.1)
rf_advanced_over_under = rf_over_under_advanced_betting(rf_total_pts_preds, rf_home_pts_preds, rf_away_pts_preds, this_season_past_games, bet_per_game_percentage=0.1, percentage_diff_to_exploit=0.05)
svm_simple_gameline = svm_gameline_simple_betting(svm_gameline_preds, this_season_past_games, bet_per_game_percentage=0.1)

#collecting and analyzing the profit from each strategy
print("random forest simple gameline betting profit: ", rf_simple_gameline[1])
print("random forest advanced gameline betting profit: ", rf_advanced_gameline[1])
print("random forest simple spread betting profit: ", rf_simple_spread[1])
print("random forest advanced spread betting profit: ", rf_advanced_spread[1])
print("random forest simple over/under betting profit: ", rf_simple_over_under[1])
print("random forest advanced over/under betting profit: ", rf_advanced_over_under[1])
print("svm simple gameline betting profit: ", svm_simple_gameline[1])

print(len(rf_simple_over_under[0]))
print(this_season_past_games.shape[0])
#creating a dataframe with all of the predictions and the game information 
betting_dict = {"rfg_preds":rf_gameline_preds, "rfgh_prob_preds":rf_gameline_prob_preds[:,1], "rfga_prob_preds":rf_gameline_prob_preds[:,0], 
                "rfh_pts_preds":rf_home_pts_preds,  "rfa_pts_preds":rf_away_pts_preds,"rfs_preds":rf_spread_preds, "rft_preds":rf_total_pts_preds, 
                "svmg_preds":svm_gameline_preds, "rfsg":rf_simple_gameline[0], "rfag":rf_advanced_gameline[0], "rfss":rf_simple_spread[0],
                "rfas":rf_advanced_spread[0], "rfsou":rf_simple_over_under[0], "rfaou":rf_advanced_over_under[0], "ssg":svm_simple_gameline[0]}
this_season_past_games2 = this_season_past_games.reset_index()
betting_df = create_betting_df(betting_dict, this_season_past_games2)
betting_df.to_csv("this_season_bets.csv")




# Putting these predictions into a betting strategy on a test set decided by the random_state value in the
# beginning of this notebook
from backtesting.predictors.RF import rf_predict_total_points, rf_predict_home_spread
from backtesting.betting.RF import (create_betting_df, rf_gameline_advanced_betting, rf_gameline_simple_betting, rf_home_spread_advanced_betting,
                        rf_home_spread_simple_betting, rf_over_under_advanced_betting, rf_over_under_simple_betting)
from backtesting.betting.SVM import svm_gameline_simple_betting

# utilizing the predictive models
rf_gameline_preds = rf_game_predictions
rf_gameline_prob_preds = rf_game_prediction_probabilities #remember this is a 2d array/list
rf_home_pts_preds = rf_predict_home_team_points(dataset_V1, random_state=random_state)[0]
rf_away_pts_preds = rf_predict_away_team_points(dataset_V1, random_state=random_state)[0]
rf_spread_preds = rf_predict_home_spread(dataset_V1, random_state=random_state)[0]
rf_total_pts_preds = rf_predict_total_points(dataset_V1, random_state=random_state)[0]
svm_gameline_preds = svm_game_predictions

# inputting predictions into betting strategy
rf_simple_gameline = rf_gameline_simple_betting(rf_gameline_preds, rfc_test_set, bet_per_game_percentage=0.1)
rf_advanced_gameline = rf_gameline_advanced_betting(rf_gameline_preds, rf_gameline_prob_preds, rfc_test_set, bet_per_game_percentage=0.1)
rf_simple_spread = rf_home_spread_simple_betting(rf_spread_preds, rfc_test_set, bet_per_game_percentage=0.1)
rf_advanced_spread = rf_home_spread_advanced_betting(rf_spread_preds, rf_home_pts_preds, rf_away_pts_preds, rfc_test_set, bet_per_game_percentage=0.1, percentage_diff_to_exploit=0.2)
rf_simple_over_under = rf_over_under_simple_betting(rf_total_pts_preds, rfc_test_set, bet_per_game_percentage=0.1)
rf_advanced_over_under = rf_over_under_advanced_betting(rf_total_pts_preds, rf_home_pts_preds, rf_away_pts_preds, rfc_test_set, bet_per_game_percentage=0.1, percentage_diff_to_exploit=0.05)
svm_simple_gameline = svm_gameline_simple_betting(svm_gameline_preds, rfc_test_set, bet_per_game_percentage=0.1)

#collecting and analyzing the profit from each strategy
print("random forest simple gameline betting profit: ", rf_simple_gameline[1])
print("random forest advanced gameline betting profit: ", rf_advanced_gameline[1])
print("random forest simple spread betting profit: ", rf_simple_spread[1])
print("random forest advanced spread betting profit: ", rf_advanced_spread[1])
print("random forest simple over/under betting profit: ", rf_simple_over_under[1])
print("random forest advanced over/under betting profit: ", rf_advanced_over_under[1])
print("svm simple gameline betting profit: ", svm_simple_gameline[1])

print(len(rf_simple_over_under[0]))
print(rfc_test_set.shape[0])
#creating a dataframe with all of the predictions and the game information 
betting_dict = {"rfg_preds":rf_gameline_preds, "rfgh_prob_preds":rf_gameline_prob_preds[:,1], "rfga_prob_preds":rf_gameline_prob_preds[:,0], 
                "rfh_pts_preds":rf_home_pts_preds,  "rfa_pts_preds":rf_away_pts_preds,"rfs_preds":rf_spread_preds, "rft_preds":rf_total_pts_preds, 
                "svmg_preds":svm_gameline_preds, "rfsg":rf_simple_gameline[0], "rfag":rf_advanced_gameline[0], "rfss":rf_simple_spread[0],
                "rfas":rf_advanced_spread[0], "rfsou":rf_simple_over_under[0], "rfaou":rf_advanced_over_under[0], "ssg":svm_simple_gameline[0]}
betting_df = create_betting_df(betting_dict, rfc_test_set)
betting_df


# Import functions for live predictions and betting
from live.predictors.RF import rf_live_predictions_master
from live.betting.RF import live_betting_simple_master, live_betting_advanced_master
live_results = rf_live_predictions_master(this_season, rfc_model, rf_predict_home_team_points(dataset_V1, random_state=random_state)[1],
                                          rf_predict_away_team_points(dataset_V1, random_state=random_state)[1], 
                                          rf_predict_total_points(dataset_V1, random_state=random_state)[1], 
                                          rf_predict_home_spread(dataset_V1, random_state=random_state)[1])


live_results

live_betting_simple_master(live_results)

live_betting_advanced_master(live_results)