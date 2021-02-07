# function to easily pull this season's gamelog. season_end_year parameter defaults to the current year
# but remember that season_end_year refers to the year in which the current regular season ends, so 
# if you're predicting a game at the beginning of a season (before the end of the year), then change this manually
# to the next year (int)
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# predict winner of unplayed games today
def rf_live_predict_winner(df, classifier, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD"):
    
    todays_games = df[df["DATE"] == todays_date]
    X = todays_games.iloc[:, todays_games.columns.get_loc(feature_start_column):todays_games.shape[1]]
    
    y_pred = classifier.predict(X)
    y_prob_pred = classifier.predict_proba(X)
    
    todays_games.insert(0, "RF_AWAY_TEAM_PROB_WIN", y_prob_pred[:,0])
    todays_games.insert(0, "RF_HOME_TEAM_PROB_WIN", y_prob_pred[:,1])
    todays_games.insert(0, "RF_HOME_WIN_PRED", y_pred)
    
    return y_pred, y_prob_pred, todays_games

# predict home team points scored of unplayed games today
def rf_live_predict_home_pts(df, regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD"):
    
    todays_games = df[df["DATE"] == todays_date]
    X = todays_games.iloc[:, todays_games.columns.get_loc(feature_start_column):todays_games.shape[1]]
    
    y_pred = regressor.predict(X)
    todays_games.insert(0, "RF_HOME_PTS_PRED", y_pred)
    
    return y_pred, todays_games

# predict away team points scored of unplaye games today
def rf_live_predict_away_pts(df, regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD"):
    
    todays_games = df[df["DATE"] == todays_date]
    X = todays_games.iloc[:, todays_games.columns.get_loc(feature_start_column):todays_games.shape[1]]
    
    y_pred = regressor.predict(X)
    todays_games.insert(0, "RF_AWAY_PTS_PRED", y_pred)
    
    return y_pred, todays_games

# predict total points scored in unplayed games today (over/under)
def rf_live_predict_total_pts(df, regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD"):
    
    todays_games = df[df["DATE"] == todays_date]
    X = todays_games.iloc[:, todays_games.columns.get_loc(feature_start_column):todays_games.shape[1]]
    
    y_pred = regressor.predict(X)
    todays_games.insert(0, "RF_TOTAL_PTS_PRED", y_pred)
    
    return y_pred, todays_games

# predict home team points spread
def rf_live_predict_home_spread(df, regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD"):
    
    todays_games = df[df["DATE"] == todays_date]
    X = todays_games.iloc[:, todays_games.columns.get_loc(feature_start_column):todays_games.shape[1]]
    
    y_pred = regressor.predict(X)
    todays_games.insert(0, "RF_HOME_SPREAD_PRED", y_pred)
    
    return y_pred, todays_games

# master function to output a summary of the game predictions for unplayed games. 
# can be used just as a table for the user to refer to when making betting decisions,
# or just out of curiosity. This will also later be used in the betting strategies module
# to make informed/optimal betting decisions for the user.
def rf_live_predictions_master(df, winner_classifier, home_pts_regressor, away_pts_regressor, 
                               total_pts_regressor,
                              home_spread_regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
                              feature_start_column="PREVIOUS_MATCHUP_RECORD"):
    
    todays_games = df[df["DATE"] == todays_date]
    master = todays_games[["HOME_TEAM", "GAME", "DATE", "AWAY_TEAM", "HOME_ODDS", "AWAY_ODDS",
                "HOME_IMPLIED_PTS", "AWAY_IMPLIED_PTS", "OVER_UNDER", "IMPLIED_HOME_SPREAD"]]
    master.insert(0, "RF_WINNER_PRED", rf_live_predict_winner(df, winner_classifier)[0])
    master.insert(0, "RF_PROB_HOME_WIN", rf_live_predict_winner(df, winner_classifier)[1][:,1])
    master.insert(0, "RF_PROB_AWAY_WIN", rf_live_predict_winner(df, winner_classifier)[1][:,0])
    master.insert(0, "RF_HOME_PTS_PRED", rf_live_predict_home_pts(df, home_pts_regressor)[0])
    master.insert(0, "RF_AWAY_PTS_PRED", rf_live_predict_away_pts(df, away_pts_regressor)[0])
    master.insert(0, "RF_TOTAL_PTS_PRED", rf_live_predict_total_pts(df, total_pts_regressor)[0])
    master.insert(0, "RF_HOME_SPREAD_PRED", rf_live_predict_home_spread(df, home_spread_regressor)[0])
    
    # make the dataset easier to understand by changing the winner column to the team predicted to win
    # and change add columns for model odds of each team to win
    winner_col = [None]*master.shape[0]
    for i, (winner, home_team, away_team) in enumerate(zip(master['RF_WINNER_PRED'], master['HOME_TEAM'], 
                                                          master['AWAY_TEAM'])):
        if winner == 'W':
            winner_col[i] = home_team
        else:
            winner_col[i] = away_team
            
    model_odds_home = [None]*master.shape[0]
    model_odds_away = [None]*master.shape[0]
    for i, (home_prob, away_prob) in enumerate(zip(master['RF_PROB_HOME_WIN'], master['RF_PROB_AWAY_WIN'])):
        model_odds_home[i] = 1/home_prob
        model_odds_away[i] = 1/away_prob
        
    master = master.drop(["RF_WINNER_PRED", "RF_PROB_HOME_WIN", "RF_PROB_AWAY_WIN"], axis=1)
    
    master.insert(master.columns.get_loc("HOME_TEAM"), "RF_MODEL_ODDS_AWAY_WIN", model_odds_away)
    master.insert(master.columns.get_loc("HOME_TEAM"), "RF_MODEL_ODDS_HOME_WIN", model_odds_home)
    master.insert(master.columns.get_loc("HOME_TEAM"), "RF_WINNER_PRED", winner_col)
    
    return master