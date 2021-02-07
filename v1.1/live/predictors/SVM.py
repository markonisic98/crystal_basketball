from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn import svm

def svm_live_predict_winner(df, classifier, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD"):
    
    todays_games = df[df["DATE"] == todays_date]
    X = todays_games.iloc[:, todays_games.columns.get_loc(feature_start_column):todays_games.shape[1]]

    y_pred = classifier.predict(X)

    todays_games.insert(0, "SVM_HOME_WIN_PRED", y_pred)
    
    return y_pred, todays_games

