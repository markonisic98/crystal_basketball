import pandas as pd
import numpy as np
from numpy import mean
from numpy import std

from sklearn.model_selection import train_test_split, RepeatedKFold, KFold
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random

# Function to add a column of 1s and 0s indicating win or loss
def make_win_loss_binary(df, win_loss_column="HOME_TEAM_WIN/LOSS", insert_placement=0):
    binary_array = [0]*df.shape[0]
    for i, each in enumerate(df[win_loss_column]):
        if each == 'W':
            binary_array[i] = 1
        else:
            binary_array[i] = 0
    try:
        df.insert(insert_placement, "HOME_TEAM_WIN/LOSS_BINARY", binary_array)
    except Exception:
        pass
    return binary_array

# Function to run an SVM model to classify a win/loss. This function has the option to take a list as input for features to be used
# This is added in the SVM function but not the RF function because the SVM model takes much longer to train, and thus it is often
# more worth it to trim a long list of features to to its most important components, accurately and quickly defined
# by running feature importance on a random forest model predicting the same column (win/loss)
# Returns an array with the win/loss predictions, and the returns the SVM model as an object, and the test set used with the random state
def svm_classify_win_loss(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_WIN/LOSS_BINARY",
                          list_important_features=[], random_state=random.randrange(1000), test_size=0.25, C=1.0,
                          kernel='rbf', gamma='scale', scale_features=False, print_vals=True):
    
    # Make win/loss column 1s and 0s if it already is not
    try:
        make_win_loss_binary(df)
    except Exception:
        print("Check win/loss column name and compare to documentation default column name.")
    # If user inputs some list to be used as most important features 
    if not list_important_features:
        X = df.iloc[:, df.columns.get_loc(feature_start_column):df.shape[1]]
    else:
        features_start = df.columns.get_loc(feature_start_column) 
        top_features = [x + features_start for x in list_important_features]
        X = df.iloc[:,top_features]
    
    y = df.iloc[:, df.columns.get_loc("HOME_TEAM_WIN/LOSS_BINARY")] # Predicting a win or a loss
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Make dataframe of the test set including features and the y value
    df.reset_index()
    test_set = pd.concat([X_test, df], axis=1, join="inner")
    test_set = test_set.loc[:,~test_set.columns.duplicated()]
    
    if scale_features:
        # Scale the features for better predictions
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    # Create a svm Classifier
    clf_svm = svm.SVC(kernel=kernel, C=C, gamma=gamma) 
        
    clf_svm.fit(X_train, y_train)
    y_pred = clf_svm.predict(X_test)

    # Measure the results 
    if print_vals:
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        print("Prediction Accuracy: ", accuracy_score(y_test, y_pred))
    
    return y_pred, clf_svm, test_set

# SVM repeated KFold test function
def svm_classify_win_loss_KFold(model, df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_WIN/LOSS_BINARY", 
                        list_important_features=[], random_state=random.randrange(1000), n_splits=10, n_repeats=2, print_vals=True):
    
    # If user inputs some list to be used as most important features 
    if not list_important_features:
            X = df.iloc[:, df.columns.get_loc(feature_start_column):df.shape[1]]
    else:
        features_start = df.columns.get_loc(feature_start_column) 
        top_features = [x + features_start for x in list_important_features]
        X = df.iloc[:,top_features]
        
    y = df.iloc[:, df.columns.get_loc("HOME_TEAM_WIN/LOSS_BINARY")]
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    i = 1
    scores = []
    for train_index, test_index in cv.split(df):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index] 
        # Train the model
        model.fit(X_train, y_train)
        print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test, model.predict(X_test))}")
        scores.append(accuracy_score(y_test, model.predict(X_test)))
        i += 1
    # Measure the results 
    if print_vals:
        print(f"Mean Accuracy Score: {mean(scores)}, Standard Deviation: {std(scores)}")
    
    return mean(scores), std(scores)

