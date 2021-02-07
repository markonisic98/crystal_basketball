import pandas as pd
import numpy as np
from numpy import mean
from numpy import std

from sklearn.model_selection import train_test_split, RepeatedKFold, KFold
# from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
#from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from matplotlib import pyplot
import random

# Random Forest Win Classification
# Takes dataframe as input, with columns named as defined in dataset builder
# Allows the option to input different column name for custom dataset, but resorts to default value
# Also takes as arguments optional random state, optional test_size split, optional min_samples_leaf, 
# optional n_jobs, and optional n_estimators (number of trees)
# Returns an array with binary classification, and probability of that classification, and the classifier model, 
# to be used in feature importance, and the test set features and y value in a dataframe to be used in betting testing
def rf_classify_win_loss(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_WIN/LOSS", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                         n_estimators=500, print_vals=False):
    
    X = df.iloc[:, df.columns.get_loc(feature_start_column):df.shape[1]]
    y = df.iloc[:, df.columns.get_loc(column_to_predict)] # Predicting a win or a loss

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)
    # Build the Tree 
    classifier = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs = n_jobs)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_prob_pred = classifier.predict_proba(X_test)
    
    # Make dataframe of the test set including features and the y value
    df.reset_index()
    test_set = pd.concat([X_test, df], axis=1, join="inner")
    test_set = test_set.loc[:,~test_set.columns.duplicated()]

    # Measure the results 
    if print_vals:
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        print("Prediction Accuracy: ", accuracy_score(y_test, y_pred))
    
    return y_pred, y_prob_pred, classifier, test_set

# Feature importance function, returns a list with the names of each column from the model ranked from most important to least
# Takes as input a predictive model and the dataframe from which it's features/prediction is based on
# Also has optional input for first feature in dataframe, assuming rest of columns past this point are features
# Also has optional input to print the float importance value for each column, or to just return the ranked column list
def feature_importance_sorted(model, df, feature_column_start="PREVIOUS_MATCHUP_RECORD", print_vals=False): 
    importance = model.feature_importances_
    important_features_dict = {}
    # Summarize feature importance
    for i,v in enumerate(importance):
        important_features_dict[i] = v
    values_sorted = sorted(important_features_dict.values(), reverse=True)
    list_sorted = sorted(important_features_dict, key=important_features_dict.get, reverse=True)
    top_columns=[None]*df.shape[0]
    for j in range(len(list_sorted)):
        top_columns[j] = df.columns[list_sorted[j]+df.columns.get_loc(feature_column_start)]
        if print_vals:
            print(top_columns[j], ": ", values_sorted[j])
    return top_columns

# Function to test a model using Repeated KFold, and returning the mean and standard deviation of these results

def rf_classify_win_loss_KFold(model, df, feature_start_column="PREVIOUS_MATCHUP_RECORD", 
                               column_to_predict="HOME_TEAM_WIN/LOSS", random_state=random.randrange(1000),
                               n_splits=10, n_repeats=2, min_samples_leaf=0.01, 
                               n_jobs=-1, n_estimators=500, print_vals=True):

    X = df.iloc[:, df.columns.get_loc(feature_start_column):df.shape[1]]
    y = df.iloc[:, df.columns.get_loc(column_to_predict)] #predicting a win or a loss

    # Build the Tree 
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    i = 1
    scores = []
    for train_index, test_index in cv.split(df):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index] 
        # Train the model
        model.fit(X_train, y_train)
        if print_vals:
            print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test, model.predict(X_test))}")
        scores.append(accuracy_score(y_test, model.predict(X_test)))
        i += 1
    # Measure the results 
    if print_vals:
        print(f"Mean Accuracy Score: {mean(scores)}, Standard Deviation: {std(scores)}")
    return mean(scores), std(scores)

# Function to predict home team pts scored in a game using random forest regression
# Returns array with predictions, mean absolute error, root mean squared error, and the regressor model
def rf_predict_home_team_points(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_PTS", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                                n_estimators=100, print_vals=True):
    X = df.iloc[:, df.columns.get_loc(feature_start_column):df.shape[1]]
    y = df.iloc[:, df.columns.get_loc(column_to_predict)] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state) 

    # Build the Tree 
    regressor = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs = n_jobs) 
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    # Make dataframe of the test set including features and the y value
    df.reset_index()
    test_set = pd.concat([X_test, df], axis=1, join="inner")
    test_set = test_set.loc[:,~test_set.columns.duplicated()]

    # Measure the results 
    if print_vals==True: 
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    return y_pred, regressor, test_set


def rf_predict_away_team_points(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="AWAY_TEAM_PTS", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                                n_estimators=100, print_vals=True):
    
    X = df.iloc[:, df.columns.get_loc(feature_start_column):df.shape[1]]
    y = df.iloc[:, df.columns.get_loc(column_to_predict)] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state) 

    # Build the Tree 
    regressor = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs = n_jobs)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    # Make dataframe of the test set including features and the y value
    df.reset_index()
    test_set = pd.concat([X_test, df], axis=1, join="inner")
    test_set = test_set.loc[:,~test_set.columns.duplicated()]

    # Measure the results 
    if print_vals==True:
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    return y_pred, regressor, test_set


def rf_predict_total_points(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="TOTAL_PTS", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                            n_estimators=100, print_vals=True):
    
    X = df.iloc[:, df.columns.get_loc(feature_start_column):df.shape[1]]
    y = df.iloc[:, df.columns.get_loc(column_to_predict)] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state) 

    # Build the Tree 
    regressor = RandomForestRegressor(n_estimators = n_estimators, min_samples_leaf = min_samples_leaf, n_jobs = n_jobs)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    # Make dataframe of the test set including features and the y value
    df.reset_index()
    test_set = pd.concat([X_test, df], axis=1, join="inner")
    test_set = test_set.loc[:,~test_set.columns.duplicated()]

    # Measure the results 
    if print_vals==True:
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    return y_pred, regressor, test_set

def rf_predict_home_spread(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_SPREAD", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                           n_estimators=100, print_vals=True):

    X = df.iloc[:, df.columns.get_loc(feature_start_column):df.shape[1]]
    y = df.iloc[:, df.columns.get_loc(column_to_predict)] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state) 

    # Build the Tree 
    regressor = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, n_jobs = n_jobs) 
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    # Make dataframe of the test set including features and the y value
    df.reset_index()
    test_set = pd.concat([X_test, df], axis=1, join="inner")
    test_set = test_set.loc[:,~test_set.columns.duplicated()]

    # Measure the results 
    if print_vals==True:
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    return y_pred, regressor, test_set

