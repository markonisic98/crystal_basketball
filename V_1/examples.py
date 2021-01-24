
from dataset_builder import multi_season_final_dataset

# Create dataset with all regular season games from the 2015/2016 season through
# to the end of the 2019/2020 season.
# This will take roughly 45 mins per season to import - by far the most computationally intensive function
# in the package (due to the scraping from various websites)
dataset_V1 = multi_season_final_dataset(2016, 2020, [5,10], 1, 82)

# Load the dataset into a csv file in this folder for easier access now and in the future
dataset_V1.to_csv("dataset_V1.csv")

# Load the dataset from the folder in a future session
import pandas as pd
dataset_V1 = pd.read_csv("dataset_V1.csv")


from random_forest_predictions import (rf_classify_win_loss, feature_importance_sorted,
                                      rf_classify_win_loss_KFold, rf_predict_home_team_points,
                                      rf_predict_away_team_points, rf_predict_total_points,
                                      rf_predict_home_spread)

# Set a random state for reproducible results through various models
# so they can be combined into one dataframe later, predicting the same games
random_state = 1111


# Predict game outcome using random forest classifier
rfc_results = rf_classify_win_loss(dataset_V1, random_state=random_state)
rf_game_predictions = rfc_results[0]
rf_game_prediction_probabilities = rfc_results[1]
rfc_model = rfc_results[2]
rfc_test_set = rfc_results[3]

# Test the accuracy of this model using repeated KFold testing with most of its
# parameters held to their default values
KFold_testing = rf_classify_win_loss_KFold(rfc_model, dataset_V1, random_state=random_state)
model_accuracy, model_std = KFold_testing[0], KFold_testing[1]
model_accuracy, model_std


from svm_predictions import svm_classify_win_loss, svm_classify_win_loss_KFold

# Predicting game outcome using SVM model
svm_results = svm_classify_win_loss(dataset_V1, C=6, random_state=random_state)
svm_game_predictions = svm_results[0]
svm_model = svm_results[1]
svm_test_set = svm_results[2]

# Test the accuracy of this model using repeated KFold testing with most of its
# parameters held to their default values
KFold_testing = svm_classify_win_loss_KFold(svm_model, dataset_V1, random_state=random_state)
model_accuracy, model_std = KFold_testing[0], KFold_testing[1]
model_accuracy, model_std


# Pulling the game lineup for a specific team on a specific night
from scrapers import get_game_lineup

get_game_lineup('BRK', '2021-01-20')


# Predicting the home team points, away team points, total points, and home team points spread in a game
from random_forest_predictions import rf_predict_total_points, rf_predict_home_spread

rfpatp = rf_predict_away_team_points(dataset_V1, random_state=random_state)
rfphtp = rf_predict_home_team_points(dataset_V1, random_state=random_state)
rfptp = rf_predict_total_points(dataset_V1, random_state=random_state)
rfphs = rf_predict_home_spread(dataset_V1, random_state=random_state)


# Putting these predictions into a betting strategy
from betting_strategies import (create_betting_df, rf_gameline_advanced_betting, rf_gameline_simple_betting, 
                                rf_home_spread_advanced_betting, rf_home_spread_simple_betting, 
                                rf_over_under_advanced_betting, rf_over_under_simple_betting, svm_gameline_simple_betting)

# utilizing the predictive models
rf_gameline_preds = rf_game_predictions
rf_gameline_prob_preds = rf_game_prediction_probabilities #remember this is a 2d array/list
rf_home_pts_preds = rfphtp[0]
rf_away_pts_preds = rfpatp[0]
rf_spread_preds = rfphs[0]
rf_total_pts_preds = rfptp[0]
svm_gameline_preds = svm_game_predictions

# inputting predictions into betting strategy
rf_simple_gameline = rf_gameline_simple_betting(rf_gameline_preds, rfc_test_set, bet_per_game_percentage=0.1)
rf_advanced_gameline = rf_gameline_advanced_betting(rf_gameline_preds, rf_gameline_prob_preds, 
                                            rfc_test_set, bet_per_game_percentage=0.1)
rf_simple_spread = rf_home_spread_simple_betting(rf_spread_preds, rfc_test_set, bet_per_game_percentage=0.1)
rf_advanced_spread = rf_home_spread_advanced_betting(rf_spread_preds, rf_home_pts_preds, rf_away_pts_preds, 
                                            rfc_test_set, bet_per_game_percentage=0.1, percentage_diff_to_exploit=0.2)
rf_simple_over_under = rf_over_under_simple_betting(rf_total_pts_preds, rfc_test_set, bet_per_game_percentage=0.1)
rf_advanced_over_under = rf_over_under_advanced_betting(rf_total_pts_preds, rf_home_pts_preds, rf_away_pts_preds, 
                                           rfc_test_set, bet_per_game_percentage=0.1, percentage_diff_to_exploit=0.05)
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

