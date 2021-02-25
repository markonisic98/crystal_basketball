# <span style="color:#0052cc"> Crystal Basketball Documentation </span> 

  

## Table of Contents
<details><summary>[backtesting](#backtesting)</summary>

 * [benchmarks](#backtesting.benchmarks)
  * []()
  * []()
  * []()
  * []()
 * [betting](#backtesting.betting)
  * [RF](#backtesting.betting.RF)
     * [create\_betting\_df](#create_betting_df)
     * [rf\_gameline\_advanced\_betting](#rf_gameline_advanced_betting)
     * [rf\_gameline\_simple\_betting](#rf_gameline_simple_betting)
     * [rf\_home\_spread\_advanced\_betting](#rf_home_spread_advanced_betting)
     * [rf\_home\_spread\_simple\_betting](#rf_home_spread_simple_betting)
     * [rf\_over\_under\_advanced\_betting](#rf_over_under_advanced_betting)
     * [rf\_over\_under\_simple\_betting](#rf_over_under_simple_betting)
  * [SVM](#backtesting.betting.SVM)
     * [create\_betting\_df](#create_betting_df)
     * [svm\_gameline\_simple\_betting](#svm_gameline_simple_betting)
 * [predictors](#backtesting.predictors)
  * [RF](#backtesting.predictors.RF)
     * [feature\_importance\_sorted](#feature_importance_sorted)
     * [rf\_classify\_win\_loss](#rf_classify_win_loss)
     * [rf\_classify\_win\_loss\_KFold](#rf_classify_win_loss_KFold)
     * [rf\_predict\_away\_team\_points](#rf_predict_away_team_points)
     * [rf\_predict\_home\_spread](#rf_predict_home_spread)
     * [rf\_predict\_home\_team\_points](#rf_predict_home_team_points)
     * [rf\_predict\_total\_points](#rf_predict_total_points)
  * [SVM](#backtesting.predictors.SVM)
     * [svm\_gameline\_simple\_betting](#svm_classify_win_loss)</details>

<details><summary>[dataCollection](#dataCollection)</summary>

 * [builder](#dataCollection.builder) 
  * [create\_final\_dataset](#create_final_dataset)
  * [get\_game\_log\_and_odds](#get_game_log_and_odds)
  * [get\_live\_season\_gamelog](#get_live_season_gamelog)
  * [multi\_season\_final\_dataset](#multi_season_final_dataset)
 * [helpers](#dataCollection.helpers)
  * [calc\_matchup\_record](#calc_matchup_record)
  * [calc\_combined\_pts\_and\_decimal\_odds](#calc_combined_pts_and_decimal_odds)
  * [create\_game\_ID](#create_game_ID)
  * [cum\_win\_loss](#cum_win_loss)
  * [gen\_moving\_avg](#gen_moving_avg)
  * [multiple\_moving\_averages](#multiple_moving_averages)
  * [win\_percentage\_moving\_avg](#win_percentage_moving_avg)
 * [scrapers](#dataCollection.scrapers)
  * [get\_game\_lineup](#get_game_lineup)
  * [get\_game\_odds](#get_game_odds)
  * [get\_team\_game\_log](#get_team_game_log)
 * [teamAbbreviations](#teamAbbreviations)</details>
 
<details><summary>[live](#live)</summary>

 * [betting](#live.betting)
  * [RF](#live.betting.RF)
     * [live\_betting\_advanced\_master](#live_betting_advanced_master)
     * [live\_betting\_simple\_master](#live_betting_simple_master)
  * [SVM](#live.betting.SVM)
 * [predictors](#live.predictors)
    * [RF](#live.predictors.RF)
      * [rf\_live\_predict\_winner](#rf_live_predict_winner)
      * [rf\_live\_predict\_away\_pts](#rf_live_predict_away_pts)
      * [rf\_live\_predict\_home\_spread](#rf_live_predict_home_spread)
      * [rf\_live\_predict\_home\_pts](#rf_live_predict_home_pts)
      * [rf\_live\_predict\_total\_pts](#rf_live_predict_total_pts)
    * [SVM](#live.predictors.SVM)
      * [svm\_live\_predict\_winner](#svm_live_predict_winner)</details>
    


## <a name="backtesting"><span style="color:#000000">backtesting</span></a>

> ## <a name="backtesting.betting"><span style="color:#404040"> betting</span></a>

>> ## <a name="backtesting.betting.RF"><span style="color:#808080"> RF</span></a>

>> #### <a name = "create_betting_df"> <span style="color:#0052cc"> create\_betting\_df </span></a> 

>> ```Python
create_betting_df(betting_strategies_dict, df)
```

>><span style="color:#000000">Creates a consolidated dataframe with the user's betting strategies, displayed in separate columns, with basic relevant game information to identify the game being played, the actual results, and the market (bookmakers') odds, over/under, and points spread. Made for better interpretability of one's betting strategy, especially compared to others. Also proves useful for creating data visualizations.
  
>> <span style="color:#000000">Returns the consolidated dataframe.  
  
>> <span style="color:#000000">Parameters: </span>  

>> * <span style="color:#000000">**betting\_strategies\_dict** (dictionary): dictionary of betting strategies, with the keys being the user's desired column name, and the values being their respective investment balance for a particular betting strategy, over a period of games (same order of games as in the dataframe inputted in the second parameter). Example: {"SVM\_simple\_betting\_strategy": [1000,900,1100,1400], "rf\_advanced\_betting\_strategy": [1000,1350,1600,2000]}. Utilize the betting strategies from this model to create the lists for the dictionary. 
>> 
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module. </span>

>>#### <a name = "rf_gameline_advanced_betting"> <span style="color:#0052cc"> rf\_gameline\_advanced\_betting </span></a> 

>>```Python
rf_gameline_advanced_betting(win_loss_preds, win_loss_prob_preds, df, investment=1000, 
                                 percentage_diff_to_exploit = 0.2, bet_per_game_percentage = 0,
                            bet_per_game_amount = 0, withdraw_at_return = 0.5, win_loss_column="HOME_TEAM_WIN/LOSS", 
                     home_odds_column="HOME_ODDS", away_odds_column="AWAY_ODDS", print_vals=False)
```
>> <span style="color:#000000">Advanced betting function for a Random Forest Classifier model, to simulate the profits that a betting strategy would return using historical data. Generally, each Machine Learning Model used will have a simple and advanced betting strategy for each of the three betting categories: gameline (win/loss), over/under (total points), and points spread (home team spread).  
In the advanced strategy, the algorithm does not focus on betting on the team that it predicts to win the game. Rather, the algorithm will choose from 3 options, based on user-defined, or default, criteria. First, the implied odds of each team winning are calculated using the probability percentage of each team winning defined by the [rf\_classify\_win\_loss](#rf_classify_win_loss) function. Next, it will check if the difference between the calculated model odds – referred to as the model implied odds – and the market odds are sufficiently different, based on criterion outlined in the *percentage\_diff\_to\_exploit* parameter. If they are sufficiently different, the model will bet on the team that has the sufficiently higher odds of winning based on model calculations. For example, if the market predicts the Atlanta Hawks to beat the Brooklyn Nets at 2.5 odds, but the model calculates their odds of winning at 1.7, a 32% difference, the function will bet the specified amount or percentage on the Atlanta Hawks to win the game. If this criterion is not met, the function will not bet on the game at all, and the investment balance after the game will remain the same as it was before the game. The logic behind this being that market odds for a team are set so that the team will win **X percent** of the time (less house cut: ~5%), derived from the market odds, in their predictions. But if the model predicts that the team will win **X + threshold percent** of the time, the AI bettor will bet on this team. The function also allows the user the option to withdraw profits at an excess return of their choice with the *withdraw\_at\_return* parameter. It is set at 0.5 (50%) by default, meaning that any time an investment reaches a balance of 50% greater than initial investment throughout a betting period of time, that extra 50% will be withdrawn and recorded as profit, and the betting will continue, starting again at the initial investment amount. The function will return the final profit, given as the amount withdrawn through the aforementioned method, and the ending investment balance over a betting period. If the user sets the value of *withdraw\_at\_return* to 0, the profit will only be the ending investment balance over the betting period.
In future versions, more advanced betting strategies will be created, betting variable amounts on each game depending on model and market predictions.
  
>><span style="color:#000000">Returns a list: **[0] = ending\_investment\_balance** and a float: **[1] = profit**, representing the investment balance after each game in the dataset, and the profit after the betting period, respectively. 
  
>><span style="color:#000000">Parameters:  

>>* <span style="color:#000000">**win\_loss\_preds** (list): a list of the win/loss predictions, each item in the list being a 'W' or an 'L'.
* <span style="color:#000000">**win\_loss\_prob\_preds** (list): a 2-D list of the probability of the home team winning/losing, gathered from the [rf\_classify\_win\_loss](#rf_classify_win_loss) function. Important to be gathered from here because this is a 2-D list that is dissected within the function to calculate model odds of each team winning, to be used in the betting strategy. 
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**investment** (float): Default = 1000. The dollar amount investment that the user wants to begin betting with. This balance will fluctuate throughout the length of the dataset as it bets (or doesn't) on each game, ultimately arriving at an ending investment balance.
* <span style="color:#000000">**percentage\_diff\_to\_exploit** (float): Default = 0.2. A decimal value representing the percentage difference in model odds vs. market odds the user designates as a requirement to meet to bet on the game.
* <span style="color:#000000">**bet\_per\_game\_percentage** (float): Default = 0. The percentage of your investment balance that you would like to bet on each game (that the algorithm decides is worth it to bet on based on your criterion). If this value is left at 0, then the **bet\_per\_game\_amount** parameter must have a positive non-zero value.
* <span style="color:#000000">**bet\_per\_game\_amount** (float): Default = 0: The amount of money you would like to bet on each game. If this value is 0, then the **bet\_per\_game\_percentage** parameter must have a positive non-zero value.
* <span style="color:#000000">**withdraw\_at\_return** (float): Default = 0.5: The percentage excess return on investment the user wants to continually withdraw at throughout the investment process. This will be kept track of and returned as a float value indicating your profit.
* <span style="color:#000000">**win\_loss\_column** (string): Default = "HOME\_TEAM\_WIN/LOSS". The name of the column in the dataframe that holds the win/loss result as "W' or 'L'.
* <span style="color:#000000">**home\_odds\_column** (string): Default = "HOME_ODDS". The name of the column in the dataframe that holds the market odds of the home team winning.
* <span style="color:#000000">**away\_odds\_column** (string): Default = "AWAY_ODDS". The name of the column in the dataframe that holds the market odds of the away team winning.
* <span style="color:#000000">**print_vals** (boolean): Default = False: If true, will print the ending investment balance list and profit float that the function also returns. Kind of redundant.

>>#### <a name = "rf_gameline_simple_betting"> <span style="color:#0052cc"> rf\_gameline\_simple\_betting </span></a> 

>>```Python
rf_gameline_simple_betting(win_loss_preds, df, investment=1000, bet_per_game_percentage = 0,
                         bet_per_game_amount = 0, withdraw_at_return = 0.5 , win_loss_column="HOME_TEAM_WIN/LOSS", 
                     home_odds_column="HOME_ODDS", away_odds_column="AWAY_ODDS", print_vals=False)
```
>><span style="color:#000000">A function to output the results of a naive gameline betting strategy based on a Random Forest Classifier imposed on historical games. The algorithm will bet on the team that it predicts to win, no matter its calculated likelihood, and take the winnings equal to the market odds multiplied by bet amount, if the team wins.

>><span style="color:#000000">Returns a list: **[0] = ending\_investment\_gbalance** and a float: **[1] = profit**, representing the investment balance after each game in the dataset, and the profit after the betting period, respectively. 
  
>><span style="color:#000000">Parameters:  

>>* <span style="color:#000000">**win\_loss\_preds** (list): a list of the win/loss predictions, each item in the list being a 'W' or an 'L'.
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**investment** (float): Default = 1000. The dollar amount investment that the user wants to begin betting with. This balance will fluctuate throughout the length of the dataset as it bets (or doesn't) on each game, ultimately arriving at an ending investment balance.
* <span style="color:#000000">**bet\_per\_game\_percentage** (float): Default = 0. The percentage of your investment balance that you would like to bet on each game (that the algorithm decides is worth it to bet on based on your criterion). If this value is left at 0, then the **bet\_per\_game\_amount** parameter must have a positive non-zero value.
* <span style="color:#000000">**bet\_per\_game\_amount** (float): Default = 0: The amount of money you would like to bet on each game. If this value is 0, then the **bet\_per\_game\_percentage** parameter must have a positive non-zero value.
* <span style="color:#000000">**withdraw\_at\_return** (float): Default = 0.5: The percentage excess return on investment the user wants to continually withdraw at throughout the investment process. This will be kept track of and returned as a float value indicating your profit.
* <span style="color:#000000">**win\_loss\_column** (string): Default = "HOME\_TEAM\_WIN/LOSS". The name of the column in the dataframe that holds the win/loss result as "W' or 'L'.
* <span style="color:#000000">**home\_odds\_column** (string): Default = "HOME_ODDS". The name of the column in the dataframe that holds the market odds of the home team winning.
* <span style="color:#000000">**away\_odds\_column** (string): Default = "AWAY_ODDS". The name of the column in the dataframe that holds the market odds of the away team winning.
* <span style="color:#000000">**print_vals** (boolean): Default = False: If true, will print the ending investment balance list and profit float that the function also returns. Kind of redundant.

>>#### <a name = "rf_home_spread_advanced_betting"> <span style="color:#0052cc"> rf\_home\_spread\_advanced\_betting </span></a> 

>>```Python
rf_home_spread_advanced_betting(home_spread_preds, home_pts_preds, away_pts_preds, df, investment=1000, 
                       bet_per_game_percentage = 0, percentage_diff_to_exploit = 0, bet_per_game_amount = 0, 
                         withdraw_at_return = 0.2, home_spread_column="HOME_TEAM_SPREAD", 
                            market_home_spread_column="IMPLIED_HOME_SPREAD", print_vals=False)
```
>><span style="color:#000000">Advanced betting function for a Random Forest Regressor model, to simulate the profits that a betting strategy would return using historical data. Generally, each Machine Learning Model used will have a simple and advanced betting strategy for each of the three betting categories: gameline (win/loss), over/under (total points), and points spread (home team spread).  
In this advanced betting strategy, the algorithm looks to exploit sufficient differences in model predicted points spread in a game, and market predicted points spread. The user defines what is considered sufficiently different in the *percentage\_diff\_to\_exploit* parameter. If it is set to 0.2, the model prediction of home team spread needs to be at least 20% higher or lower than the market points spread to bet either side. As an additional restrictive criterion, the algorithm will only decide to bet on a games points spread if the home points spread prediction and the individual home points less away points predictions all agree with each other. If the guess is correct, the winning odds are set at 1.91, to account for a typical house cut.
In future versions, more advanced betting strategies will be created, betting variable amounts on each game depending on model and market predictions.
  
>><span style="color:#000000">Returns a list: **[0] = ending\_investment\_balance** and a float: **[1] = profit**, representing the investment balance after each game in the dataset, and the profit after the betting period, respectively. 
  
>><span style="color:#000000">Parameters:  

>>* <span style="color:#000000"> **home\_spread\_preds** (list): a list of the points spread (from the home team perspective) predictions, each item in the list being a float value.
* <span style="color:#000000">**home\_pts\_preds** (list): a list of the home points predictions, each item in the list being a float value.
* <span style="color:#000000">**away\_pts\_preds** (list): a list of the away points predictions, each item in the list being a float value.
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**investment** (float): Default = 1000. The dollar amount investment that the user wants to begin betting with. This balance will fluctuate throughout the length of the dataset as it bets (or doesn't) on each game, ultimately arriving at an ending investment balance.
* <span style="color:#000000">**percentage\_diff\_to\_exploit** (float): Default = 0. A decimal value representing the percentage difference in model odds vs. market odds the user designates as a requirement to meet to bet on the game.
* <span style="color:#000000">**bet\_per\_game\_percentage** (float): Default = 0. The percentage of your investment balance that you would like to bet on each game (that the algorithm decides is worth it to bet on based on your criterion). If this value is left at 0, then the **bet\_per\_game\_amount** parameter must have a positive non-zero value.
* <span style="color:#000000">**bet\_per\_game\_amount** (float): Default = 0: The amount of money you would like to bet on each game. If this value is 0, then the **bet\_per\_game\_percentage** parameter must have a positive non-zero value.
* <span style="color:#000000">**withdraw\_at\_return** (float): Default = 0.2: The percentage excess return on investment the user wants to continually withdraw at throughout the investment process. This will be kept track of and returned as a float value indicating your profit.
* <span style="color:#000000">**home\_spread\_column** (string): Default = "HOME_TEAM_SPREAD". The name of the column in the dataframe pertaining the actual home team points spread in a game.
* <span style="color:#000000">**market\_home\_spread\_column** (string): Default = "IMPLIED_HOME_SPREAD". The name of the column in the dataframe that holds the points spread for a game as set out by the market.
* <span style="color:#000000">**print_vals** (boolean): Default = False: If true, will print the ending investment balance list and profit float that the function also returns. Kind of redundant.

>>#### <a name = "rf_home_spread_simple_betting"> <span style="color:#0052cc"> rf\_home\_spread\_simple\_betting </span></a> 

>>```Python
rf_home_spread_simple_betting(home_spread_preds, df, investment=1000, bet_per_game_percentage = 0, 
                                percentage_diff_to_exploit = 0, bet_per_game_amount = 0, withdraw_at_return = 0.2,
                                 home_spread_column="HOME_TEAM_SPREAD", 
                            market_home_spread_column="IMPLIED_HOME_SPREAD", print_vals=False)
```
>><span style="color:#000000">Simple function to bet on the home team points spread of a given game, using the Random Forest Regressor model predictions of home spread in a game. Trivially, if the model predicts the home spread to be higher than the market home spread, then the algorithm will bet on the over, and if it predicts it to be lower, then vice versa. Winning odds are set at 1.91 to account for house cut. The function also allows for the *percentage\_dif\_to\_exploit* parameter, as a restrictive criterion for betting, to (hopefully) maximize profits or minimize risk. The difference between the simple and advanced strategy in this case is that the advanced strategy has the additional restrictive criterion that the home points spread model prediction has to agree with the individual home and away points predictions, relative to the market home points spread, for a bet to be placed.
  
>><span style="color:#000000">Returns a list: **[0] = ending\_investment\_balance** and a float: **[1] = profit**, representing the investment balance after each game in the dataset, and the profit after the betting period, respectively. 
  
>><span style="color:#000000">Parameters:  

>>* <span style="color:#000000">**home\_spread\_preds** (list): a list of the points spread (from the home team perspective) predictions, each item in the list being a float value.
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**investment** (float): Default = 1000. The dollar amount investment that the user wants to begin betting with. This balance will fluctuate throughout the length of the dataset as it bets (or doesn't) on each game, ultimately arriving at an ending investment balance.
* <span style="color:#000000">**percentage\_diff\_to\_exploit** (float): Default = 0. A decimal value representing the percentage difference in model odds vs. market odds the user designates as a requirement to meet to bet on the game.
* <span style="color:#000000">**bet\_per\_game\_percentage** (float): Default = 0. The percentage of your investment balance that you would like to bet on each game (that the algorithm decides is worth it to bet on based on your criterion). If this value is left at 0, then the **bet\_per\_game\_amount** parameter must have a positive non-zero value.
* <span style="color:#000000">**bet\_per\_game\_amount** (float): Default = 0: The amount of money you would like to bet on each game. If this value is 0, then the **bet\_per\_game\_percentage** parameter must have a positive non-zero value.
* <span style="color:#000000">**withdraw\_at\_return** (float): Default = 0.2: The percentage excess return on investment the user wants to continually withdraw at throughout the investment process. This will be kept track of and returned as a float value indicating your profit.
* <span style="color:#000000">**home\_spread\_column** (string): Default = "HOME_TEAM_SPREAD". The name of the column in the dataframe pertaining the actual home team points spread in a game.
* <span style="color:#000000">**market\_home\_spread\_column** (string): Default = "IMPLIED_HOME_SPREAD". The name of the column in the dataframe that holds the points spread for a game as set out by the market.
* <span style="color:#000000">**print_vals** (boolean): Default = False: If true, will print the ending investment balance list and profit float that the function also returns. Kind of redundant.

>>#### <a name = "rf_over_under_advanced_betting"> <span style="color:#0052cc"> rf\_over\_under\_advanced\_betting </span></a> 

>>```Python
rf_over_under_advanced_betting(total_pts_preds, home_pts_preds, away_pts_preds, df, investment=1000, 
                                   bet_per_game_percentage = 0, percentage_diff_to_exploit = 0, 
                                   bet_per_game_amount = 0, withdraw_at_return = 0.2,
                                 total_pts_column="TOTAL_PTS", 
                            over_under_column="OVER_UNDER", print_vals=False)
```
>><span style="color:#000000">Advanced betting function for a Random Forest Regressor model, to simulate the profits that a betting strategy would return using historical data. Generally, each Machine Learning Model used will have a simple and advanced betting strategy for each of the three betting categories: gameline (win/loss), over/under (total points), and points spread (home team spread).  
In this advanced betting strategy, the algorithm looks to exploit sufficient differences in model predicted total points scored in a game, and market predicted total points (over/under). The user defines what is considered sufficiently different in the *percentage\_diff\_to\_exploit* parameter. If it is set to 0.2, the model prediction of total points needs to be at least 20% higher or lower than the market over/under to bet either side. If it is left at 0, it is equivalent to the [rf\_over\_under\_simple\_betting](#rf_over_under_simple_betting) strategy. As an additional restrictive criterion, the algorithm will only choose to bet on a game if the total points prediction **and** the home points + away points individual predictions are on the same side of the over/under. If the guess is correct, the winning odds are set at 1.91, to account for a typical house cut.
In future versions, more advanced betting strategies will be created, betting variable amounts on each game depending on model and market predictions.
  
>><span style="color:#000000">Returns a list: **[0] = ending\_investment\_balance** and a float: **[1] = profit**, representing the investment balance after each game in the dataset, and the profit after the betting period, respectively. 
  
>><span style="color:#000000">Parameters:  

>>* <span style="color:#000000">**total\_pts\_preds** (list): a list of the total points predictions, each item in the list being a float value.
* <span style="color:#000000">**home\_pts\_preds** (list): a list of the home points predictions, each item in the list being a float value.
* <span style="color:#000000">**away\_pts\_preds** (list): a list of the away points predictions, each item in the list being a float value.
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**investment** (float): Default = 1000. The dollar amount investment that the user wants to begin betting with. This balance will fluctuate throughout the length of the dataset as it bets (or doesn't) on each game, ultimately arriving at an ending investment balance.
* <span style="color:#000000">**percentage\_diff\_to\_exploit** (float): Default = 0.2. A decimal value representing the percentage difference in model odds vs. market odds the user designates as a requirement to meet to bet on the game.
* <span style="color:#000000">**bet\_per\_game\_percentage** (float): Default = 0. The percentage of your investment balance that you would like to bet on each game (that the algorithm decides is worth it to bet on based on your criterion). If this value is left at 0, then the **bet\_per\_game\_amount** parameter must have a positive non-zero value.
* <span style="color:#000000">**bet\_per\_game\_amount** (float): Default = 0: The amount of money you would like to bet on each game. If this value is 0, then the **bet\_per\_game\_percentage** parameter must have a positive non-zero value.
* <span style="color:#000000">**withdraw\_at\_return** (float): Default = 0.2: The percentage excess return on investment the user wants to continually withdraw at throughout the investment process. This will be kept track of and returned as a float value indicating your profit.
* <span style="color:#000000">**total\_pts\_column** (string): Default = "TOTAL_PTS". The name of the column in the dataframe pertaining the actual total points scored in a game.
* <span style="color:#000000">**over\_under\_column** (string): Default = "OVER_UNDER". The name of the column in the dataframe that holds the over/under for a game as set out by the market.
* <span style="color:#000000">**print_vals** (boolean): Default = False: If true, will print the ending investment balance list and profit float that the function also returns. Kind of redundant.


>>#### <a name = "rf_over_under_simple_betting"> <span style="color:#0052cc"> rf\_over\_under\_simple\_betting </span></a> 

>>```Python
rf_over_under_simple_betting(total_pts_preds, df, investment=1000, bet_per_game_percentage = 0, 
                                percentage_diff_to_exploit = 0, bet_per_game_amount = 0, withdraw_at_return = 0.2,
                                 total_pts_column="TOTAL_PTS", 
                            over_under_column="OVER_UNDER", print_vals=False)
```
>><span style="color:#000000">Simple function to bet on the over/under of a given game, using the Random Forest Regressor model predictions of total points scored in a game. Trivially, if the model predicts the total points to be higher than the market over/under, then the algorithm will bet on the over, and if it predicts it to be lower, then vice versa. Winning odds are set at 1.91 to account for house cut. The function also allows for the *percentage\_dif\_to\_exploit* parameter, as a restrictive criterion for betting, to (hopefully) maximize profits or minimize risk. The difference between the simple and advanced strategy in this case is that the advanced strategy has the additional restrictive criterion that the total points model prediction has to agree with the individual home and away points predictions, relative to the market over/under, for a bet to be placed.
  
>><span style="color:#000000">Returns a list: **[0] = ending\_investment\_balance** and a float: **[1] = profit**, representing the investment balance after each game in the dataset, and the profit after the betting period, respectively. 
  
>><span style="color:#000000">Parameters:  

>>* <span style="color:#000000">**total\_pts\_preds** (list): a list of the total points predictions, each item in the list being a float value.
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**investment** (float): Default = 1000. The dollar amount investment that the user wants to begin betting with. This balance will fluctuate throughout the length of the dataset as it bets (or doesn't) on each game, ultimately arriving at an ending investment balance.
* <span style="color:#000000">**percentage\_diff\_to\_exploit** (float): Default = 0.2. A decimal value representing the percentage difference in model odds vs. market odds the user designates as a requirement to meet to bet on the game.
* <span style="color:#000000">**bet\_per\_game\_percentage** (float): Default = 0. The percentage of your investment balance that you would like to bet on each game (that the algorithm decides is worth it to bet on based on your criterion). If this value is left at 0, then the **bet\_per\_game\_amount** parameter must have a positive non-zero value.
* <span style="color:#000000">**bet\_per\_game\_amount** (float): Default = 0: The amount of money you would like to bet on each game. If this value is 0, then the **bet\_per\_game\_percentage** parameter must have a positive non-zero value.
* <span style="color:#000000">**withdraw\_at\_return** (float): Default = 0.2: The percentage excess return on investment the user wants to continually withdraw at throughout the investment process. This will be kept track of and returned as a float value indicating your profit.
* <span style="color:#000000">**total\_pts\_column** (string): Default = "TOTAL_PTS". The name of the column in the dataframe pertaining the actual total points scored in a game.
* <span style="color:#000000">**over\_under\_column** (string): Default = "OVER_UNDER". The name of the column in the dataframe that holds the over/under for a game as set out by the market.
* <span style="color:#000000">**print_vals** (boolean): Default = False: If true, will print the ending investment balance list and profit float that the function also returns. Kind of redundant.

>> ## <a name="backtesting.betting.SVM"><span style="color:#808080"> SVM</span></a>

>> #### <a name = "create_betting_df"> <span style="color:#0052cc"> create\_betting\_df </span></a> 

>> ```Python
create_betting_df(betting_strategies_dict, df)
```

>><span style="color:#000000">Creates a consolidated dataframe with the user's betting strategies, displayed in separate columns, with basic relevant game information to identify the game being played, the actual results, and the market (bookmakers') odds, over/under, and points spread. Made for better interpretability of one's betting strategy, especially compared to others. Also proves useful for creating data visualizations.
  
>> <span style="color:#000000">Returns the consolidated dataframe.  
  
>> <span style="color:#000000">Parameters: </span>  

>> * <span style="color:#000000">**betting\_strategies\_dict** (dictionary): dictionary of betting strategies, with the keys being the user's desired column name, and the values being their respective investment balance for a particular betting strategy, over a period of games (same order of games as in the dataframe inputted in the second parameter). Example: {"SVM\_simple\_betting\_strategy": [1000,900,1100,1400], "rf\_advanced\_betting\_strategy": [1000,1350,1600,2000]}. Utilize the betting strategies from this model to create the lists for the dictionary. 
>> 
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module. </span>

>>#### <a name = "svm_gameline_simple_betting"> <span style="color:#0052cc"> svm\_gameline\_simple\_betting </span></a> 

>>```Python
svm_gameline_simple_betting(win_loss_preds, df, investment=1000, bet_per_game_percentage = 0,
                        bet_per_game_amount = 0, withdraw_at_return = 0.5, win_loss_column="HOME_TEAM_WIN/LOSS", 
                     home_odds_column="HOME_ODDS", away_odds_column="AWAY_ODDS", print_vals=False)
```
>><span style="color:#000000">A function to output the results of a naive gameline betting strategy based on a Support Vector Machine Classifier imposed on historical games. The algorithm will bet on the team that it predicts to win, no matter its calculated likelihood, and take the winnings equal to the market odds multiplied by bet amount, if the team wins.

>><span style="color:#000000">Returns a list: **[0] = ending\_investment\_gbalance** and a float: **[1] = profit**, representing the investment balance after each game in the dataset, and the profit after the betting period, respectively. 
  
>><span style="color:#000000">Parameters:  

>>* <span style="color:#000000">**win\_loss\_preds** (list): a list of the win/loss predictions, each item in the list being an binary integer, either 0 or 1.
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**investment** (float): Default = 1000. The dollar amount investment that the user wants to begin betting with. This balance will fluctuate throughout the length of the dataset as it bets (or doesn't) on each game, ultimately arriving at an ending investment balance.
* <span style="color:#000000">**bet\_per\_game\_percentage** (float): Default = 0. The percentage of your investment balance that you would like to bet on each game (that the algorithm decides is worth it to bet on based on your criterion). If this value is left at 0, then the **bet\_per\_game\_amount** parameter must have a positive non-zero value.
* <span style="color:#000000">**bet\_per\_game\_amount** (float): Default = 0: The amount of money you would like to bet on each game. If this value is 0, then the **bet\_per\_game\_percentage** parameter must have a positive non-zero value.
* <span style="color:#000000">**withdraw\_at\_return** (float): Default = 0.5: The percentage excess return on investment the user wants to continually withdraw at throughout the investment process. This will be kept track of and returned as a float value indicating your profit.
* <span style="color:#000000">**win\_loss\_column** (string): Default = "HOME\_TEAM\_WIN/LOSS". The name of the column in the dataframe that holds the win/loss result as "W' or 'L'.
* <span style="color:#000000">**home\_odds\_column** (string): Default = "HOME_ODDS". The name of the column in the dataframe that holds the market odds of the home team winning.
* <span style="color:#000000">**away\_odds\_column** (string): Default = "AWAY_ODDS". The name of the column in the dataframe that holds the market odds of the away team winning.
* <span style="color:#000000">**print_vals** (boolean): Default = False: If true, will print the ending investment balance list and profit float that the function also returns. Kind of redundant.

> ## <a name="backtesting.predictors"><span style="color:#404040"> predictors</span></a>

>> ## <a name="backtesting.predictors.RF"><span style="color:#808080"> RF</span></a>

>>#### <a name = "feature_importance_sorted"> <span style="color:#0052cc"> feature\_importance_sorted </span></a> 

>>```Python
feature_importance_sorted(model, df, feature_column_start="PREVIOUS_MATCHUP_RECORD", print_vals=False)
```
>><span style="color:#000000">Creates a list containing the names of the most important features (columns) used in building a predictive model, sorted from most important to least important.
  
>><span style="color:#000000">Returns a sorted list of the most important features used in building the predictive model inputted as a parameter.
  
>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**model** (ML model object): a predictive model, as an object. 
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**feature\_column_start** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
* <span style="color:#000000">**print_vals** (boolean): Default = True. If true, this will print the entire list of the sorted features, and their respective importance as a float value (essentially the percentage of model accuracy explained by each feature).

>>#### <a name = "rf_classify_win_loss"> <span style="color:#0052cc"> rf\_classify\_win_loss </span></a> 

>>```Python
rf_classify_win_loss(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_WIN/LOSS", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                         n_estimators=500, print_vals=False)
```
>><span style="color:#000000">Function to simplify the process of running a Random Forest Classifier model on a dataset. Though if the user is looking for more customizability in their Random Forest Classifier, we suggest using the original **RandomForestClassifier** function from the **sklearn.ensemble** package.  This function splits the inputted dataset into a dataset with features, and a dataset with the column to be predicted. It then splits these datasets further into a test set, and a training set used to build the Random Forest Classifier model to predict the outcome of games. More information on the Random Forest Classifier used, and Random Forest models in general, can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). The parameters will be explained in more detail below, but it is important to note that if the user wants to build a dataframe with multiple separate predictions, ammend the *random_state* parameter from the function to an integer consistent among all functions at play. This will ensure that the dataset is split into training and test sets using the same indices, so predictions are consistent. This is especially important for testing betting strategies with multiple predictions at once.
  
>><span style="color:#000000">Returns:  
 
>>* <span style="color:#000000">2 lists: **[0] = y_pred, [1] = y\_prob\_pred** - an array with the win/loss prediction for each game, as a 'W' or 'L' string, and a 2-dimensional   list holding the probability of the home team winning and the probability of the away team winning, respectively.
* <span style="color:#000000">1 object: **[2] = classifier** - the classifier object built in the function.
* <span style="color:#000000">1 dataframe: **[3] = test_set** - a dataframe with all of the same columns/contents as the original dataframe, but with only the test set indices created from the train/test split in the function.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
* <span style="color:#000000">**column\_to_predict** (string): Default = "HOME\_TEAM\_WIN/LOSS" (the column name created when using the [builder](#dataCollection.builder) module). Name of the column being predicted by the Random Forest classifier model.
* <span style="color:#000000">**random_state** (int): Default: random int between 1 and 1000. A random state will create reproducible results with respect to the splitting of the test and training dataset.
* <span style="color:#000000">**test_size** (float): Default = 0.25. Represents the percentage of data dedicated to the test set, and the remainder is reserved for the training set. Generally, you want to keep the test set size smaller than the training set size.
* <span style="color:#000000">**min\_samples\_leaf** (float): Default = 0.01. Represents the percentage of samples in the training set that must fall into a leaf node for the random forest model for it to be added as a node into the tree. This is one way to prevent overfitting in the model. More information on this can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
* <span style="color:#000000">**n_jobs** (int): Default = -1. The number of jobs to run in parallel. -1 means using all processors.
* <span style="color:#000000">**n_estimators** (int): Default = 500. The number of trees in the random forest. Anything above several hundred generally produces negligible accuracy improvement.
* <span style="color:#000000">**print_vals** (boolean): Default = False. If true, function will print the confusion matrix for the predictions, the classification report, and the model prediction accuracy, as a percentage.

>>#### <a name = "rf_classify_win_loss_KFold"> <span style="color:#0052cc"> rf\_classify\_win\_loss_KFold </span></a> 

>>```Python
rf_classify_win_loss_KFold(model, df, feature_start_column="PREVIOUS_MATCHUP_RECORD", 
                               column_to_predict="HOME_TEAM_WIN/LOSS", random_state=random.randrange(1000),
                               n_splits=10, n_repeats=2, min_samples_leaf=0.01, 
                               n_jobs=-1, n_estimators=500, print_vals=True)
```
>><span style="color:#000000">Function to simplify testing a predictive model using repeated KFold testing (can read about this method more [here](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/)). This offers a quick way of assessing one models predictive power over another's, or just to see generally how well your model predicts something, and with what variance (standard deviation in this case).
  
>><span style="color:#000000">Returns 2 float values: **[0] = mean(scores), [1] = std(scores)**, representing the mean accuracy score and standard deviation of the accuracy for the model from the repeated testing, respectively.

>><span style="color:#000000">Parameters:  

>>* <span style="color:#000000">**model** (ML model object): a predictive model, as an object. In this case, a Random Forest Classifier model.  
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
* <span style="color:#000000">**column\_to_predict** (string): Default = "HOME\_TEAM\_WIN/LOSS" (the column name created when using the [builder](#dataCollection.builder) module). Name of the column being predicted by the Random Forest Classifier model.
* <span style="color:#000000">**random_state** (int): Default: random int between 1 and 1000. A random state will create reproducible results with respect to the splitting of the test and training dataset.
* <span style="color:#000000">**n_splits** (int): Default = 10. Number of segments the K-Fold test splits the dataset into. Also, the K-Fold test is repeated **n_splits** times.
* <span style="color:#000000">**n_repeats** (int): Default = 2. Number of times the K-Fold process as a whole is repeated.
* <span style="color:#000000">**min\_samples\_leaf** (float): Default = 0.01. Represents the percentage of samples in the training set that must fall into a leaf node for the random forest model for it to be added as a node into the tree. This is one way to prevent overfitting in the model. More information on this can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
* <span style="color:#000000">**n_jobs** (int): Default = -1. The number of jobs to run in parallel. -1 means using all processors.
* <span style="color:#000000">**n_estimators** (int): Default = 500. The number of trees in the random forest. Anything above several hundred generally produces negligible accuracy improvement.
* <span style="color:#000000">**print_vals** (boolean): Default = False. If true, function will print the accuracy of each fold (Default: 20 Folds (10 x 2)), to provide more detail and live results.

>>#### <a name = "rf_predict_away_team_points"> <span style="color:#0052cc"> rf\_predict\_away\_team\_points </span></a> 

>>```Python
rf_predict_away_team_points(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="AWAY_TEAM_PTS", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                                n_estimators=100, print_vals=True)
```
>><span style="color:#000000">Function to simplify the process of running a Random Forest Regressor model on a dataset. In this case: used to predict the number of points scored by the away team. Uses the dataset created in the [builder](#dataCollection.builder) module. Uses the **Random Forest Regressor** function from the **sklearn.ensemble** package. Splits the dataset into a training set and a dataset - the proportions in each set specified by the user.

>><span style="color:#000000">Returns: 
 
>>* <span style="color:#000000">1 list: **[0] = y\_pred** - a list of the predictions for the away team points scored for each game   
>>* <span style="color:#000000">1 object: **[1] = regressor** - the Random Forest Regressor model for away team points as an object   
>>* <span style="color:#000000">1 dataframe: **[2] = test_set** - a dataframe with all of the same columns/contents as the original dataframe, but with only the test set indices created from the train/test split in the function.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
* <span style="color:#000000">**column\_to_predict** (string): Default = "AWAY\_TEAM\_PTS" (the column name created when using the [builder](#dataCollection.builder) module). Name of the column being predicted by the Random Forest Regressor model.
* <span style="color:#000000">**random_state** (int): Default: random int between 1 and 1000. A random state will create reproducible results with respect to the splitting of the test and training dataset.
* <span style="color:#000000">**test_size** (float): Default = 0.25. Represents the percentage of data dedicated to the test set, and the remainder is reserved for the training set. Generally, you want to keep the test set size smaller than the training set size.
* <span style="color:#000000">**min\_samples\_leaf** (float): Default = 0.01. Represents the percentage of samples in the training set that must fall into a leaf node for the random forest model for it to be added as a node into the tree. This is one way to prevent overfitting in the model. More information on this can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
* <span style="color:#000000">**n_jobs** (int): Default = -1. The number of jobs to run in parallel. -1 means using all processors.
* <span style="color:#000000">**n_estimators** (int): Default = 100. The number of trees in the random forest. Anything above several hundred generally produces negligible accuracy improvement.
* <span style="color:#000000">**print_vals** (boolean): Default = False. If true, function will print the mean absolute error, mean squared error, and root mean squared error of the predictions.

>>#### <a name = "rf_predict_home_spread"> <span style="color:#0052cc"> rf\_predict\_home\_spread </span></a> 

>>```Python
rf_predict_home_spread(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_SPREAD", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                           n_estimators=100, print_vals=True)
```
>><span style="color:#000000">Function to simplify the process of running a Random Forest Regressor model on a dataset. In this case: used to predict the points spread, from the perspective of the home team, in a game. As a reminder, the spread for a Team A in a game is defined as Team B's points less Team A's points. If Team A wins the game by 10 points, the spread for Team A is -10  Uses the dataset created in the [builder](#dataCollection.builder) module. Uses the **Random Forest Regressor** function from the **sklearn.ensemble** package. Splits the dataset into a training set and a dataset - the proportions in each set specified by the user.

>><span style="color:#000000">Returns: 
 
>>* <span style="color:#000000">1 list: **[0] = y\_pred** - a list of the predictions for the home team spread for each game   
* <span style="color:#000000">1 object: **[1] = regressor** - the Random Forest Regressor model for home team spread as an object   
* <span style="color:#000000">1 dataframe: **[2] = test_set** - a dataframe with all of the same columns/contents as the original dataframe, but with only the test set indices created from the train/test split in the function.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
* <span style="color:#000000">**column\_to_predict** (string): Default = "HOME\_TEAM\_SPREAD" (the column name created when using the [builder](#dataCollection.builder) module). Name of the column being predicted by the Random Forest Regressor model.
* <span style="color:#000000">**random_state** (int): Default: random int between 1 and 1000. A random state will create reproducible results with respect to the splitting of the test and training dataset.
* <span style="color:#000000">**test_size** (float): Default = 0.25. Represents the percentage of data dedicated to the test set, and the remainder is reserved for the training set. Generally, you want to keep the test set size smaller than the training set size.
* <span style="color:#000000">**min\_samples\_leaf** (float): Default = 0.01. Represents the percentage of samples in the training set that must fall into a leaf node for the random forest model for it to be added as a node into the tree. This is one way to prevent overfitting in the model. More information on this can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
* <span style="color:#000000">**n_jobs** (int): Default = -1. The number of jobs to run in parallel. -1 means using all processors.
* <span style="color:#000000">**n_estimators** (int): Default = 100. The number of trees in the random forest. Anything above several hundred generally produces negligible accuracy improvement.
* <span style="color:#000000">**print_vals** (boolean): Default = False. If true, function will print the mean absolute error, mean squared error, and root mean squared error of the predictions.

>>#### <a name = "rf_predict_home_team_points"> <span style="color:#0052cc"> rf\_predict\_home\_team\_points </span></a> 

>>```Python
rf_predict_home_team_points(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_PTS", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                                n_estimators=100, print_vals=True)
```
>><span style="color:#000000">Function to simplify the process of running a Random Forest Regressor model on a dataset. In this case: used to predict the number of points scored by the home team. Uses the dataset created in the [builder](#dataCollection.builder) module. Uses the **Random Forest Regressor** function from the **sklearn.ensemble** package. Splits the dataset into a training set and a dataset - the proportions in each set specified by the user.

>><span style="color:#000000">Returns:  

>>* <span style="color:#000000">1 list: **[0] = y\_pred** - a list of the predictions for the home team points scored for each game   
* <span style="color:#000000">1 object: **[1] = regressor** - the Random Forest Regressor model for home team points as an object   
* <span style="color:#000000">1 dataframe: **[2] = test_set** - a dataframe with all of the same columns/contents as the original dataframe, but with only the test set indices created from the train/test split in the function.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
* <span style="color:#000000">**column\_to_predict** (string): Default = "HOME\_TEAM\_PTS" (the column name created when using the [builder](#dataCollection.builder) module). Name of the column being predicted by the Random Forest Regressor model.
* <span style="color:#000000">**random_state** (int): Default: random int between 1 and 1000. A random state will create reproducible results with respect to the splitting of the test and training dataset.
* <span style="color:#000000">**test_size** (float): Default = 0.25. Represents the percentage of data dedicated to the test set, and the remainder is reserved for the training set. Generally, you want to keep the test set size smaller than the training set size.
* <span style="color:#000000">**min\_samples\_leaf** (float): Default = 0.01. Represents the percentage of samples in the training set that must fall into a leaf node for the random forest model for it to be added as a node into the tree. This is one way to prevent overfitting in the model. More information on this can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
* <span style="color:#000000">**n_jobs** (int): Default = -1. The number of jobs to run in parallel. -1 means using all processors.
* <span style="color:#000000">**n_estimators** (int): Default = 100. The number of trees in the random forest. Anything above several hundred generally produces negligible accuracy improvement.
* <span style="color:#000000">**print_vals** (boolean): Default = False. If true, function will print the mean absolute error, mean squared error, and root mean squared error of the predictions.

>>#### <a name = "rf_predict_total_points"> <span style="color:#0052cc"> rf\_predict\_total\_points </span></a> 

>>```Python
rf_predict_total_points(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="TOTAL_PTS", 
                        random_state=random.randrange(1000), test_size=0.25, min_samples_leaf=0.01, n_jobs=-1, 
                            n_estimators=100, print_vals=True)
```
>><span style="color:#000000">Function to simplify the process of running a Random Forest Regressor model on a dataset. In this case: used to predict the number of total points scored in a game. Uses the dataset created in the [builder](#dataCollection.builder) module. Uses the **Random Forest Regressor** function from the **sklearn.ensemble** package. Splits the dataset into a training set and a dataset - the proportions in each set specified by the user.

>><span style="color:#000000">Returns: 
 
>>* <span style="color:#000000">1 list: **[0] = y\_pred** - a list of the predictions for the total points scored for each game   
* <span style="color:#000000">1 object: **[1] = regressor** - the Random Forest Regressor model for total points as an object   
* <span style="color:#000000">1 dataframe: **[2] = test_set** - a dataframe with all of the same columns/contents as the original dataframe, but with only the test set indices created from the train/test split in the function.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
* <span style="color:#000000">**column\_to_predict** (string): Default = "TOTAL_PTS" (the column name created when using the [builder](#dataCollection.builder) module). Name of the column being predicted by the Random Forest Regressor model.
* <span style="color:#000000">**random_state** (int): Default: random int between 1 and 1000. A random state will create reproducible results with respect to the splitting of the test and training dataset.
* <span style="color:#000000">**test_size** (float): Default = 0.25. Represents the percentage of data dedicated to the test set, and the remainder is reserved for the training set. Generally, you want to keep the test set size smaller than the training set size.
* <span style="color:#000000">**min\_samples\_leaf** (float): Default = 0.01. Represents the percentage of samples in the training set that must fall into a leaf node for the random forest model for it to be added as a node into the tree. This is one way to prevent overfitting in the model. More information on this can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
* <span style="color:#000000">**n_jobs** (int): Default = -1. The number of jobs to run in parallel. -1 means using all processors.
* <span style="color:#000000">**n_estimators** (int): Default = 100. The number of trees in the random forest. Anything above several hundred generally produces negligible accuracy improvement.
* <span style="color:#000000">**print_vals** (boolean): Default = False. If true, function will print the mean absolute error, mean squared error, and root mean squared error of the predictions.

>> ## <a name="backtesting.predictors.SVM"><span style="color:#808080">SVM</span></a>

>>#### <a name = "make_win_loss_binary"> <span style="color:#0052cc"> make\_win\_loss\_binary </span></a> 

>>```Python
make_win_loss_binary(df, win_loss_column="HOME_TEAM_WIN/LOSS", insert_placement=0)
```
>><span style="color:#000000">Helper function to make the win/loss column for a dataframe, typically consisting of 'W' or 'L', a binary integer column, consisting of 1 or 0. 1 indicates a win for the home team, and 0 a loss. The function then inserts the column into the inputted dataframe in the desired position.
  
>><span style="color:#000000">Returns a binary list/array consisting of 1s and 0s in the same order of the win/loss column of the inputted dataframe. 
  
>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**win\_loss_column** (string): Default: "HOME\_TEAM\_WIN/LOSS". The win/loss column of the dataframe.
* <span style="color:#000000">**insert_placement** (integer): Default: 0. Represents the index at which the user wants to insert the binary win loss column.

>>#### <a name = "svm_classify_win_loss"> <span style="color:#0052cc"> svm\_classify\_win\_loss </span></a> 

>>```Python
svm_classify_win_loss(df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_WIN/LOSS_BINARY",
                          list_important_features=[], random_state=random.randrange(1000), test_size=0.25, C=0, kernel='rbf',
                           gamma='scale', scale_features=False, print_vals=True)
```
>><span style="color:#000000">Function to predict the outcome of a game using a Support Vector Machine (SVM) classifier. In a very basic sense, an SVM model creates a hyperplane that is positioned optimally in between data points in two different classes, to best separate them. These datapoints are positioned based on their feature values - in this case, hundreds of features. This function uses an *Radial Basis Function* kernel SVM model. More information on how SVM models work and how they can be customized can be found [here](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47).
  
>><span style="color:#000000">Returns:  
  
>>* <span style="color:#000000">**[0] = y_pred** - a list of the classification predictions of the SVM classifier, based on the test set created in the function.
* <span style="color:#000000">**[1] = clf_svm** - the SVM classifier model as an object
* <span style="color:#000000">**[2] = test_set** - the dataframe used as the test set for the SVM classifier, with the indices selected using the train test split method, with users able to create reproducible indices by assigning a consistent integer value to the *random_state* parameter.
  
>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
* <span style="color:#000000">**column\_to_predict** (string): Default = "HOME\_TEAM\_WIN/LOSS\_BINARY" (the column name created when using the [builder](#dataCollection.builder) module). Name of the column being predicted by the SVM Classifier.
* <span style="color:#000000">**list\_important\_features** (list): an optional list containing the columns to be used in the building of the SVM Classifier. This is best and easiest done by running a Random Forest Classifier on the dataset to predict a win or a loss, and then inputting that into the [feature\_importance\_sorted](#feature_importance_sorted) function. The resulting list returned from this function can be inputted into this parameter up until any number of top features is reached, as desired by the user (e.g., splicing the top 150 features in the list [:150]).
* <span style="color:#000000">**random_state** (int): Default: random int between 1 and 1000. A random state will create reproducible results with respect to the splitting of the test and training dataset.
* <span style="color:#000000">**test_size** (float): Default = 0.25. Represents the percentage of data dedicated to the test set, and the remainder is reserved for the training set. Generally, you want to keep the test set size smaller than the training set size.
* <span style="color:#000000">**C** (int): Default = 1.0. Tells the SVM Classifier how much you want to avoid misclassifying each training example. A better explanation can be found in [this thread](https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel) 
* <span style="color:#000000">**kernel** (string): Default = 'rbf'. Kernel to be used in the SVM model. Short paper on kernel selection and feature scaling decisions can be found [here](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)
* <span style="color:#000000">**gamma** (string or float): Default = 'scale'. Gamma to be used in the SVM model. Can be a float value, or it can be 'auto' or 'scale'. Short paper involving kernel selection and gamma  can be found [here](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)
* <span style="color:#000000">**scale_features** (boolean): Default = False. If True, will scale (normalize) the model features. This may only prove useful in certain instances. Short paper on kernel selection and feature scaling decisions can be found [here](https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf)
* <span style="color:#000000">**print_vals** (boolean): Default = False. If true, function will print the confusion matrix, classification report, and prediction accuracy.

>>#### <a name = "svm_classify_win_loss_KFold"> <span style="color:#0052cc"> svm\_classify\_win\_loss\_KFold </span></a> 

>>```Python
svm_classify_win_loss_KFold(model, df, feature_start_column="PREVIOUS_MATCHUP_RECORD", column_to_predict="HOME_TEAM_WIN/LOSS_BINARY", 
                        list_important_features=[], random_state=random.randrange(1000), n_splits=10, n_repeats=2, print_vals=True)
```
>><span style="color:#000000">Function to simplify testing a predictive model using repeated KFold testing (can read about this method more [here](https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/)). This offers a quick way of assessing one models predictive power over another's, or just to see generally how well your model predicts something, and with what variance (standard deviation in this case).
  
>><span style="color:#000000">Returns 2 float values: **[0] = mean(scores), [1] = std(scores)**, representing the mean accuracy score and standard deviation of the accuracy for the model from the repeated testing, respectively.

>><span style="color:#000000">Parameters:  

>>* <span style="color:#000000">**model** (ML model object): a predictive model, as an object. In this case, an SVM Classifier model.  
* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
* <span style="color:#000000">**list\_important\_features** (list): an optional list containing the columns to be used in the building of the SVM Classifier. This is best and easiest done by running a Random Forest Classifier on the dataset to predict a win or a loss, and then inputting that into the [feature\_importance\_sorted](#feature_importance_sorted) function. The resulting list returned from this function can be inputted into this parameter up until any number of top features is reached, as desired by the user (e.g., splicing the top 150 features in the list [:150]).
* <span style="color:#000000">**random_state** (int): Default: random int between 1 and 1000. A random state will create reproducible results with respect to the splitting of the test and training dataset.
* <span style="color:#000000">**n_splits** (int): Default = 10. Number of segments the K-Fold test splits the dataset into. Also, the K-Fold test is repeated **n_splits** times.
* <span style="color:#000000">**n_repeats** (int): Default = 2. Number of times the K-Fold process as a whole is repeated.
* <span style="color:#000000">**print_vals** (boolean): Default = True. If true, function will print the accuracy of each fold (Default: 20 Folds (10 x 2)), to provide more detail and live results.

## <a name="dataCollection"><span style="color:#000000">dataCollection</span></a>

>## <a name="dataCollection.builder"><span style="color:#404040">builder</span></a>

>#### <a name = "create_final_dataset"> <span style="color:#0052cc"> create\_final_dataset </span></a> 

>```Python
create_final_dataset(season_end_year, moving_avgs_list, first_game_to_collect=1, last_game_to_collect=82)
```
><span style="color:#000000">Creates a league-wide dataset for each season's games' stats and results, using the moving/cumulative averages defined in [get\_game\_log\_and_odds](#get_game_log_and_odds). Organizes and renames the columns for consistencies, and consolidates the data in general for a cleaned dataset to be used in the training of a model.  
  
><span style="color:#000000">Returns the cleaned dataset.  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**season\_end_year** (int): the year in which the season in question ends (e.g., 2016 represents the 2015/2016 NBA season).
* <span style="color:#000000">**moving\_avgs_list** (list): a list containing the desired moving averages to be collected for each game log stat (e.g., [3,5,10] will collect the 3,5, and 10 game moving averages for each stat. The program also collects the cumulative average for each stat by default).
* <span style="color:#000000">**first\_game\_to_collect** (int): Default: 1. Specifies from which game of the season the user wants to start collecting data. If the user wants to collect a full season's data, this value should remain at its default. This parameter is offered because it may prove easier to train a model on games happening around the middle of the season – exclusive of the tail-ends due to factors such as small sample size, load management, and loss of motivation.
* <span style="color:#000000">**last\_game\_to_collect** (int): Default: 82. Specifies the last game of the season the user wants to collect data from for each team. If the user wants to collect data for the whole season, this value should remain at its default. This parameter is offered because it may prove easier to train a model on games happening around the middle of the season – exclusive of the tail-ends due to factors such as small sample size, load management, and loss of motivation.


>#### <a name = "get_game_log_and_odds"> <span style="color:#0052cc"> get\_game\_log\_and_odds </span></a> 

>```Python
get_game_log_and_odds(team, season_end_year, moving_avgs_list, print_progress=True)
```

><span style="color:#000000">Merges a single team/season game log and the matching odds (gameline, over/under, spread) for each game. Utilizes the [get\_team\_game_log](#get_team_game_log) and [get\_game_odds](#get_game_odds) functions from the [scrapers](#scrapers) module.   
  
><span style="color:#000000">Returns the merged dataframe.

><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**team** (string): team name (abbreviation) (e.g., BOS)
* <span style="color:#000000">**season\_end_year** (int): the year in which the season in question ends (e.g., 2016 represents the 2015/2016 NBA season)
* <span style="color:#000000">**moving\_avgs_list** (list): a list containing the desired moving averages to be collected for each game log stat (e.g., [3,5,10] will collect the 3,5, and 10 game moving averages for each stat. The program also collects the cumulative average for each stat by default).
* <span style="color:#000000">**print_progress** (boolean): Default: True. Prints the progress of the function when it is called if True, showing you which team/season has been loaded - to show that the web scraper is in fact pulling data. Default value is True.

>#### <a name = "get_live_season_gamelog"> <span style="color:#0052cc"> get\_live\_season\_gamelog </span></a> 

>```Python
get_live_season_gamelog(moving_avgs_list, season_end_year=int(datetime.today().strftime('%Y')))
```
><span style="color:#000000">Creates a dataframe including all games that have occured in the upcoming season, and their respective features/results, as well as games that are to occur on the same day. Utilizes the [create\_final\_dataset](#create_final_dataset) function
  
><span style="color:#000000">Returns the season's gamelog in a dataframe.
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**moving\_avgs\_list** (list): A list containing the various moving average lengths the user would like to collect for each stat. Commonly, the user would collect [5,10] game moving averages


  
>#### <a name = "multi_season_final_dataset"> <span style="color:#0052cc"> multi\_season\_final_dataset </span></a> 

>```Python
multi_season_final_dataset(season_end_year_start, season_end_year_end, moving_avgs_list, 
                               first_game_to_collect=1, last_game_to_collect=82)
```
><span style="color:#000000">Creates a final dataset of season statistics using multiple seasons' worth of data. Essentially just runs the [create\_final_dataset](#create_final_dataset) function through a loop for each desired season and appends the data into one final dataframe.  
  
><span style="color:#000000">Returns the cleaned dataset.  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**season\_end\_year_start** (int): the year in which the first season desired ends (e.g., 2016 represents the 2015/2016 NBA season).
* <span style="color:#000000">**season\_end\_year_end** (int): the year in which the last season desired ends (e.g., 2016 represents the 2015/2016 NBA season).
* <span style="color:#000000">**moving\_avgs_list** (list): a list containing the desired moving averages to be collected for each game log stat (e.g., [3,5,10] will collect the 3,5, and 10 game moving averages for each stat. The program also collects the cumulative average for each stat by default).
* <span style="color:#000000">**first\_game\_to_collect** (int): Default: 1. Specifies from which game of the season the user wants to start collecting data. If the user wants to collect a full season's data, this value should remain at its default. This parameter is offered because it may prove easier to train a model on games happening around the middle of the season – exclusive of the tail-ends due to factors such as small sample size, load management, and loss of motivation.
* <span style="color:#000000">**last\_game\_to_collect** (int): Default: 82. Specifies from which game of the season the user wants to start collecting data. If the user wants to collect a full season's data, this value should remain at its default. This parameter is offered because it may prove easier to train a model on games happening around the middle of the season – exclusive of the tail-ends due to factors such as small sample size, load management, and loss of motivation.  



>## <a name="dataCollection.helpers"><span style="color:#404040">helpers</span></a>

>#### <a name = "calc_matchup_record"> <span style="color:#0052cc"> calc\_matchup_record </span></a> 

>```Python
calc_matchup_record(df)
```
><span style="color:#000000">Function to create a new derivative statistic, encompassing a team's record against another team. We chose not to use win percentage against another team in the same season because that would equalize the win percentage of a team who has not played another team yet with a team who has only lost to another team they have played (0%). Thus, we created a stat named *PREVIOUS\_MATCHUP_RECORD*, which is an integer value. Each team's stat in this category will be equal and opposite to the opposing team's value in this stat. It begins at 0, and for each win that Team A gets against Team B, their *PREVIOUS\_MATCHUP_RECORD* will go up by 1, while Team B's *PREVIOUS\_MATCHUP_RECORD* goes down by 1. Because a team only plays another team in the NBA a maximum of 4 times in a season, and this stat represents all the games between two teams previous to their current matchup, this value falls in the range of integers from [-3,3] \(inclusive).
  
><span style="color:#000000">Returns a list with the matchup record for each game from the perspective of the home team in the game in question.  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.

>#### <a name = "calc_combined_pts_and_decimal_odds"> <span style="color:#0052cc"> calc\_combined\_pts\_and\_decimal_odds </span></a> 

>```Python
calc_combined_pts_and_decimal_odds(df)
```
><span style="color:#000000">Three-in-one function. First, calculates the total points in a game, by summing the points that each team scored. Second, calculates the home team spread from these same two points columns. Third, converts the American odds scraped from the internet into decimal odds for easier calculations/comparisons in future modules/functions.
  
><span style="color:#000000">Returns 4 lists: **[0] = total\_pts, [1] = actual\_home\_spread, [2] = home\_odds, [3] = away_odds**. These are used to ammend the final dataset in the [builder](#dataCollection.builder) module.  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.

>#### <a name = "create_game_ID"> <span style="color:#0052cc"> create\_game_ID </span></a> 

>```Python
create_game_ID(df)
```
><span style="color:#000000">Creates a game ID to be used as the ID/index for each game in a dataset. Creates a string by combining the date of the game played with each team's abbreviation, sorted alphabetically to keep consistent when merging datasets. This ensures a unique ID for every NBA game (e.g., *2021-01-16BRKORL* represents the game played between the Brooklyn Nets and the Orlando Magic on January 16, 2021).  
  
><span style="color:#000000">Returns the game ID as a string  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe with *TEAM*, *OPP*, and *DATE* columns, created using the [builder](#dataCollection.builder) module.

>#### <a name = "cum_win_loss"> <span style="color:#0052cc"> cum\_win_loss </span></a> 

>```Python
cum_win_loss(df)
```
><span style="color:#000000">Calculates the cumulative number of wins a team has before each game. Used as a helper function in the [win\_percentage\_moving_avg](#win_percentage_moving_avg) function.
  
><span style="color:#000000">Returns a list with the cumulative wins before each game in the dataset.  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe with a *WIN/LOSS* column, created using the [builder](#dataCollection.builder) module.

>#### <a name = "gen_moving_avg"> <span style="color:#0052cc"> gen\_moving_avg </span></a> 

>```Python
gen_moving_avg(array, len_moving_avg)
```
><span style="color:#000000">Calculates the specified moving average of a stat before each game from a given array containing the stat's value in each game.
  
><span style="color:#000000">Returns a list with the moving average of the inputted stat.  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**array** (list/array): an array containing the stat for which the user would like to collect a moving average for before each game. In this case, the array inputted is a dataframe column representing a gamelog stat for each game.
>* <span style="color:#000000">**len\_moving_avg** (int): specified the number of games the user wants to calculate a moving average for, prior to the current game.

>#### <a name = "multiple_moving_averages"> <span style="color:#0052cc"> multiple\_moving_averages </span></a> 

>```Python
multiple_moving_averages(df, list_moving_avgs)
```
><span style="color:#000000">Calculates any amount of moving averages specified or each stat in a team's game log. Utilizes the [gen\_moving_avg](#gen_moving_avg) and [win\_percentage\_moving_avg](#win_percentage_moving_avg) functions, looping through all of the collected stats in the dataframe.
  
><span style="color:#000000">Returns a new dataframe with new columns created for each desired moving average of each stat.  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**df** (pandas dataframe): the initial partially-cleaned dataframe created from the [get\_team\_game_log](#get_team_game_log) function in the [scrapers](#scrapers) module. 
* <span style="color:#000000">**list\_moving_avgs** (list): a list containing the desired moving averages to be collected for each game log stat (e.g., [3,5,10] will collect the 3,5, and 10 game moving averages for each stat. The program also collects the cumulative average for each stat by default). moving\_avgs\_list and list\_moving_avgs is used interchangeably throughout the project.

>#### <a name = "win_percentage_moving_avg"> <span style="color:#0052cc"> win\_percentage\_moving_avg </span></a> 

>```Python
win_percentage_moving_avg(df, len_moving_avg)
```
><span style="color:#000000">Calculates the win percentage of a team at each game in their season, specified by a moving average length. The moving average length parameter tells the function for which period of games, prior to the current one, they would like to collect a win percentage for. Utilizes [cum\_win_loss](#cum_win_loss) function.
  
><span style="color:#000000">Returns a list with the win percentage at each game for the specified number of games prior to the current one.  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
>* <span style="color:#000000">**len\_moving_avg** (int): specified the number of games the user wants to calculate a win percentage for, prior to the current game.

>## <a name="dataCollection.scrapers"><span style="color:#404040">scrapers</span></a>

>#### <a name = "get_game_lineup"> <span style="color:#0052cc"> get\_game\_lineup </span></a> 

>```Python
get_game_lineup(team, date)
```
><span style="color:#000000">Scrapes active player lineup for a team in a game from [roto grinders](https://rotogrinders.com/). When the roster is not completely confirmed before a game, the lineup is projected rather than confirmed. The function makes no distinction between the two, for simplicity, but a majority of the lineups are confirmed rather than projected. This will be used to add individual player statistics to model predictions of game outcomes, and (hopefully) fill in the prediction gaps that are created when a team has injuries, roster changes, or is resting players. 
  
><span style="color:#000000">Returns a list of the active players for a team for a specific game.  
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**team** (string): takes the 3 letter abbreviation for the NBA team for which the user desires a roster lineup.
* <span style="color:#000000">**date** (string): takes a date as a string, in the form of 'YYYY-MM-DD', for the date of the game the user desires a roster lineup.

>#### <a name = "get_game_odds"> <span style="color:#0052cc"> get\_game\_odds </span></a> 

>```Python
get_game_odds(team, date)
```
><span style="color:#000000">Scrapes the odds and other betting information for any NBA game desired past January 2015. Betting information includes: market (bookmaker's) odds of each team winning (inherently accounts for a house cut), market (bookmaker's) points spread for each team, and market (bookmaker's) over/under. Although very extremely rare, there is occasionally no betting information for a game, and in one instance (so far) there is absurdly extreme odds for one team to win/lose, which is likely some error in the time they collected the market odds (right before a team was about to win), or a typo. In these cases, we add neutral odds that account for a house cut: 1.91 (-110) odds for each team to win, 0 home team spread, and 200 over/under. Though there are certainly better ways of accounting for these cases, they are far too rare to add any complex mechanisms to solve them.
  
><span style="color:#000000">Returns a list of length 6, comprised of the following information:  
  
>* <span style="color:#000000">**[0] = team\_implied\_pts** - the implied team points based on over/under and points spread 
* <span style="color:#000000">**[1] = team_odds** - the market odds for the team to win, in American form.
* <span style="color:#000000">**[2] = over_under** - the market over/under
* <span style="color:#000000">**[3] = team_spread** - the market spread for the team in question
* <span style="color:#000000">**[4] = opp\_implied\_pts** - the implied opponent points based on over/under and points spread
* <span style="color:#000000">**[5] = opp odds** - the market odds for the opponent to win, in American form.
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**team** (string): takes the 3 letter abbreviation for the NBA team for which the user desires betting information.
* <span style="color:#000000">**date** (string): takes a date as a string, in the form of 'YYYY-MM-DD', for the date of the game the user desires betting information.

>#### <a name = "get_team_game_log"> <span style="color:#0052cc"> get\_team\_game\_log </span></a> 

>```Python
get_team_game_log(team, season_end_year, moving_avg_list, add_upcoming=False,                                date_upcoming=datetime.today().strftime('%Y-%m-%d'))
```
><span style="color:#000000">Scrapes the game log statistics for each game in a season for a team from [basketball-reference.com](https://www.basketball-reference.com/), and transforms the stats into moving averages lagging one period behind each game, to be used as features in training predictive models. The number of moving averages, and the number of games they encompass, are chosen by the user and entered as a list (e.g., [3,5,10] will collect the 3-game, 5-game, and 10-game moving averages for each stat in a team's gamelog). The cumulative average for each stat is collected by default, up until the game *before* the game in which the cumulative value of a stat is displayed. Also has the option to add an upcoming game, to be used in a "live" prediction later, rather than for backtesting purposes.
  
><span style="color:#000000">Returns a dataframe consisting of all of the gamelog stats and information for a team, including columns inserted for the moving averages for each stat. Dataframe returned with an upcoming game added will have empty values in the win/loss, home team points, and away team points columns, as it has not been played yet.
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**team** (string): takes the 3 letter abbreviation for the NBA team for which the user desires gamelog stats.
* <span style="color:#000000">**season\_end_year** (int): the year in which the season in question ends (e.g., 2016 represents the 2015/2016 NBA season)
* <span style="color:#000000">**moving\_avgs_list** (list): a list containing the desired moving averages to be collected for each game log stat (e.g., [3,5,10] will collect the 3,5, and 10 game moving averages for each stat. The program also collects the cumulative average for each stat by default).
* <span style="color:#000000">**add_upcoming** (boolean): Default = False. If true, the program will add an upcoming game to the dataframe, based on the date parameter.
* <span style="color:#000000">**date_upcoming** (string): Default = today's date. Date must be later than yesterday. The date of the upcoming game the user would like to collect info for, to be used in a "live" prediction, rather than backtesting. It is best used on the day of the upcoming game, as the lineups will be known, and the moving average stats will be the most up-to-date. A user can also add a game far into the future, but this will offer little predictive power as there are many games in between that will affect the averages used as features in predictions.

>#### <a name = "get_upcoming_game"> <span style="color:#0052cc"> get\_upcoming\_game</span></a> 

>```Python
get_upcoming_game(team, date = datetime.today().strftime('%Y-%m-%d'))
```
><span style="color:#000000">Helper function. Creates an upcoming game row with some empty fields to be added in the [get\_team\_game\_log](#get_team_game_log) function. This can then be used to make a "live" prediction on a game, rather than only backtesting predictions on games that have already been played.
  
><span style="color:#000000">Returns a dataframe with one row, with the date of the game, game number, home team, away team, with the index as the game_ID created by using the [create\_game\_ID](#create_game_ID) function.
  
><span style="color:#000000">Parameters:  
  
>* <span style="color:#000000">**team** (string): takes the 3 letter abbreviation for the NBA team for which the user desires gamelog stats.
* <span style="color:#000000">**date_upcoming** (string): Default = today's date. Date must be later than yesterday. The date of the upcoming game the user would like to collect info for, to be used in a "live" prediction, rather than backtesting. It is best used on the day of the upcoming game, as the lineups will be known, and the moving average stats will be the most up-to-date. A user can also add a game far into the future, but this will offer little predictive power as there are many games in between that will affect the averages used as features in predictions.

>## <a name="teamAbbreviations"><span style="color:#404040">teamAbbreviations</span></a>

><span style="color:#000000">Python module/file containg only a dictionary with all 30 NBA teams and their respective team abbreviations as per [basketball-reference](https://www.basketball-reference.com/). Note that there are discrepancies in two of these team abbreviations between websites: *'BRK'* is sometimes referred to as *'BKN'*, and *'CHO'* is sometimes referred to as *'CHA'*. Anywhere that this may cause errors in merging dataframes, most notably in the [scrapers](#scrapers) module, this has been accounted for within the function.

>```Python
TEAM_TO_TEAM_ABBR = {
        'ATLANTA HAWKS': 'ATL',
        'BOSTON CELTICS': 'BOS',
        'BROOKLYN NETS': 'BRK',
        'CHICAGO BULLS': 'CHI',
        'CHARLOTTE HORNETS': 'CHO',
        'CLEVELAND CAVALIERS': 'CLE',
        'DALLAS MAVERICKS': 'DAL',
        'DENVER NUGGETS': 'DEN',
        'DETROIT PISTONS': 'DET',
        'GOLDEN STATE WARRIORS': 'GSW',
        'HOUSTON ROCKETS': 'HOU',
        'INDIANA PACERS': 'IND',
        'LOS ANGELES CLIPPERS': 'LAC',
        'LOS ANGELES LAKERS': 'LAL',
        'MEMPHIS GRIZZLIES': 'MEM',
        'MIAMI HEAT': 'MIA',
        'MILWAUKEE BUCKS': 'MIL',
        'MINNESOTA TIMBERWOLVES': 'MIN',
        'NEW ORLEANS PELICANS' : 'NOP',
        'NEW YORK KNICKS' : 'NYK',
        'OKLAHOMA CITY THUNDER' : 'OKC',
        'ORLANDO MAGIC' : 'ORL',
        'PHILADELPHIA 76ERS' : 'PHI',
        'PHOENIX SUNS' : 'PHO',
        'PORTLAND TRAIL BLAZERS' : 'POR',
        'SACRAMENTO KINGS' : 'SAC',
        'SAN ANTONIO SPURS' : 'SAS',
        'TORONTO RAPTORS' : 'TOR',
        'UTAH JAZZ' : 'UTA',
        'WASHINGTON WIZARDS' : 'WAS'
}   
```

## <a name="live"><span style="color:#404040">live</span></a>

> ## <a name="live.betting"><span style="color:#404040"> betting</span></a>

>> ## <a name="live.betting.RF"><span style="color:#808080"> RF</span></a>

>> #### <a name = "live_betting_simple_master"> <span style="color:#0052cc"> live\_betting\_simple\_master </span></a> 

>> ```Python
live_betting_simple_master(df, table_form=True, text_form=False, visual_form=False)
```

>><span style="color:#000000">Creates a master dataframe indicating the model's betting decisions for NBA games today. This function employs a "simple" betting strategy, requiring no threshold likelihood/prediction difference between model and market prediction to recommend a bet. Takes as input a consolidated and cleaned dataframe created using the  [rf\_live\_predictions\_master](#rf_live_predictions_master) from the [live.predictors.RF](#live.predictors.RF) module. Offers 1 form to return betting strategy for today's games: a table. Will eventually be able to return strategies in text form (sentences), and in visual form, with team logos and fun things like that :-). 
  
>> <span style="color:#000000">Returns the consolidated dataframe with betting strategy reccomendations for each game of the day.  
  
>> <span style="color:#000000">Parameters: </span>  
 
>> * <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [rf\_live\_predictions\_master](#rf_live_predictions_master) module. </span>
* <span style="color:#000000">**table_form** (boolean): Default: True. If true, will return the day's betting reccomendations in the form of a  table. Currently the only form available to return, will eventually support text form and visual form.
* <span style="color:#000000">**text_form** (boolean): Default: False. If true, will return the day's betting reccomendations in text form (sentences). Currently not available.
* <span style="color:#000000">**visual_form** (boolean): Default: False. If true, will return the day's betting reccomendations in visual form (logos and pictures). Currently not available.

>> #### <a name = "live_betting_advanced_master"> <span style="color:#0052cc"> live\_betting\_advanced\_master </span></a> 

>> ```Python
live_betting_advanced_master(df, odds_threshold=0.2, over_under_threshold=0.05, spread_threshold=0.4,
                                table_form=True, text_form=False, visual_form=False)
```

>><span style="color:#000000">Creates a master dataframe indicating the model's betting decisions for NBA games today. This function employs an "advanced" betting strategy, requiring a threshold likelihood/prediction difference between model and market prediction to recommend a bet. This threshold difference is in the form of a percentage - a percentage difference in model predictions vs. market predictions. This offers bets that are more sufficiently more likely to hit than the market suggests, to offer a higher expected value per bet for the user. Default values are assigned to these thresholds that we currently recommend (minimal testing). Subject to change by author recommendation or user preferences. For spread predictions and total points predictions (over/under), the individual model predictions of each team's points less the other's must generally agree with the holisitc prediction - either spread or over/under - for the model to recommend a bet in the advanced strategy. Takes as input a consolidated and cleaned dataframe created using the [rf\_live\_predictions\_master](#rf_live_predictions_master) from the [live.predictors.RF](#live.predictors.RF) module. Offers 1 form to return betting strategy for today's games: a table. Will eventually be able to return strategies in text form (sentences), and in visual form, with team logos and fun things like that :-). 
  
>> <span style="color:#000000">Returns the consolidated dataframe with betting strategy reccomendations for each game of the day.  
  
>> <span style="color:#000000">Parameters: </span>  
 
>> * <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [rf\_live\_predictions\_master](#rf_live_predictions_master) module. </span>
* <span style="color:#000000">**odds_threshold** (float): Default = 0.2. A percentage difference required between the market odds and model odds for the model to recommend the user to bet on one team or the other.
* <span style="color:#000000">**over_under_threshold** (float): Default = 0.05. A percentage difference in over/under predictions between the model and market for the model to recommend for the user to bet over or under.
* <span style="color:#000000">**spread_threshold** (float): Default = 0.4. A percentage difference between the model's point spread prediction and market point spread prediction to recommend a bet. 
* <span style="color:#000000">**table_form** (boolean): Default: True. If true, will return the day's betting reccomendations in the form of a  table. Currently the only form available to return, will eventually support text form and visual form.
* <span style="color:#000000">**text_form** (boolean): Default: False. If true, will return the day's betting reccomendations in text form (sentences). Currently not available.
* <span style="color:#000000">**visual_form** (boolean): Default: False. If true, will return the day's betting reccomendations in visual form (logos and pictures). Currently not available.

>> ## <a name="live.betting.SVM"><span style="color:#808080"> SVM</span></a>
>> Currently not available

> ## <a name="live.predictors"><span style="color:#404040">predictors</span></a>

>> ## <a name="live.predictors.RF"><span style="color:#808080"> RF</span></a>

>>#### <a name = "rf_live_predict_winner"> <span style="color:#0052cc"> rf\_live\_predict\_winner </span></a> 

>>```Python
rf_live_predict_winner(df, classifier, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD")
```
>><span style="color:#000000">Function to simplify predicting the winner of a game that is yet to occur. Utilizes the *RandomForestClassifer* function, and the moving averages/cumulative average of each feature before the game to predict the game. Takes as input a cleaned and consolidated dataframe created from the [create\_final\_dataset](#create_final_dataset) or [multi\_season\_dataset](#multi_season_final_dataset) functions.
  
>><span style="color:#000000">Returns:  
 
>>* <span style="color:#000000">2 lists: **[0] = y_pred, [1] = y\_prob\_pred** - an array with the win/loss prediction for each game, as a 'W' or 'L' string, and a 2-dimensional list holding the probability of the home team winning and the probability of the away team winning, respectively.
* <span style="color:#000000">1 object: **[2] = todays_games** - the dataset including all features and (empty) results columns for the games happening today.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**classifier** (RandomForestClassifier object). The model created from using historical data from the [backtesting](#backtesting) module and the [rf\_classify\_win\_loss](#rf_classify_win_loss) function. This model is used to predict the outcome of today's games.
* <span style="color:#000000">**todays_date** (string): Default: today's date, using the *datetime* package. Today's date, in the form of 'YYYY-MM-DD'. Leave this as default, unless you want to predict the next days game's at the end of the current day (rather than the default of predicting on the day of). Also consider manually changing if there are timezone differences.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.

>>#### <a name = "rf_live_predict_away_pts"> <span style="color:#0052cc"> rf\_live\_predict\_away\_pts </span></a> 

>>```Python
rf_live_predict_away_pts(df, regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD")
```
>><span style="color:#000000">Function to simplify the process of running a Random Forest Regressor model on a dataset. In this case: used to predict the number of points scored by the away team. Takes as input a dataset created in the [builder](#dataCollection.builder) module of the current season's games. 

>><span style="color:#000000">Returns: 
 
>>* <span style="color:#000000">1 list: **[0] = y\_pred** - a list of the predictions for the away team points scored for each game     
>>* <span style="color:#000000">1 dataframe: **[1] = todays_games** - a dataframe with all of the features, information, and (empty) game results columns of todays games.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**regressor** (RandomForestRegressor object): RandomForestRegressor model as an object, returned when using the [rf\_predict\_away\_team\_points](#rf_predict_away_team_points) function in the [backtesting](#backtesting) module. This is used to make the actual predictions on today's games
* <span style="color:#000000">**todays_date** (string): Default: today's date, using the *datetime* package. Today's date, in the form of 'YYYY-MM-DD'. Leave this as default, unless you want to predict the next days game's at the end of the current day (rather than the default of predicting on the day of). Also consider manually changing if there are timezone differences.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.

>>#### <a name = "rf_live_predict_home_spread"> <span style="color:#0052cc"> rf\_live\_predict\_home\_spread</span></a> 

>>```Python
rf_live_predict_home_spread(df, regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD")
```
>><span style="color:#000000">Function to simplify the process of running a Random Forest Regressor model on a dataset. In this case: used to predict the points spread in a game from the perspective of the home team. Takes as input a dataset created in the [builder](#dataCollection.builder) module of the current season's games. 

>><span style="color:#000000">Returns: 
 
>>* <span style="color:#000000">1 list: **[0] = y\_pred** - a list of the predictions for the points spread for each game     
>>* <span style="color:#000000">1 dataframe: **[1] = todays_games** - a dataframe with all of the features, information, and (empty) game results columns of todays games.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**regressor** (RandomForestRegressor object): RandomForestRegressor model as an object, returned when using the [rf\_predict\_home\_spread](#rf_predict_home_spread) function in the [backtesting](#backtesting) module. This is used to make the actual predictions on today's games
* <span style="color:#000000">**todays_date** (string): Default: today's date, using the *datetime* package. Today's date, in the form of 'YYYY-MM-DD'. Leave this as default, unless you want to predict the next days game's at the end of the current day (rather than the default of predicting on the day of). Also consider manually changing if there are timezone differences.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.

>>#### <a name = "rf_live_predict_home_pts"> <span style="color:#0052cc"> rf\_live\_predict\_home\_pts </span></a> 

>>```Python
rf_live_predict_home_pts(df, regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD")
```
>><span style="color:#000000">Function to simplify the process of running a Random Forest Regressor model on a dataset. In this case: used to predict the number of points scored by the home team. Takes as input a dataset created in the [builder](#dataCollection.builder) module of the current season's games. 

>><span style="color:#000000">Returns: 
 
>>* <span style="color:#000000">1 list: **[0] = y\_pred** - a list of the predictions for the home team points scored for each game     
>>* <span style="color:#000000">1 dataframe: **[1] = todays_games** - a dataframe with all of the features, information, and (empty) game results columns of todays games.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**regressor** (RandomForestRegressor object): RandomForestRegressor model as an object, returned when using the [rf\_predict\_home\_team\_points](#rf_predict_home_team_points) function in the [backtesting](#backtesting) module. This is used to make the actual predictions on today's games
* <span style="color:#000000">**todays_date** (string): Default: today's date, using the *datetime* package. Today's date, in the form of 'YYYY-MM-DD'. Leave this as default, unless you want to predict the next days game's at the end of the current day (rather than the default of predicting on the day of). Also consider manually changing if there are timezone differences.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.

>>#### <a name = "rf_live_predict_total_pts"> <span style="color:#0052cc"> rf\_live\_predict\_total\_pts </span></a> 

>>```Python
rf_live_predict_total_pts(df, regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD")
```
>><span style="color:#000000">Function to simplify the process of running a Random Forest Regressor model on a dataset. In this case: used to predict the total number of points scored in a game. Takes as input a dataset created in the [builder](#dataCollection.builder) module of the current season's games. 

>><span style="color:#000000">Returns: 
 
>>* <span style="color:#000000">1 list: **[0] = y\_pred** - a list of the predictions for the home team points scored for each game     
>>* <span style="color:#000000">1 dataframe: **[1] = todays_games** - a dataframe with all of the features, information, and (empty) game results columns of todays games.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**regressor** (RandomForestRegressor object): RandomForestRegressor model as an object, returned when using the [rf\_predict\_total\_points](#rf_predict_total_points) function in the [backtesting](#backtesting) module. This is used to make the actual predictions on today's games
* <span style="color:#000000">**todays_date** (string): Default: today's date, using the *datetime* package. Today's date, in the form of 'YYYY-MM-DD'. Leave this as default, unless you want to predict the next days game's at the end of the current day (rather than the default of predicting on the day of). Also consider manually changing if there are timezone differences.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.

>>#### <a name = "rf_live_predictions_master"> <span style="color:#0052cc"> rf\_live\_predictions\_master </span></a> 

>>```Python
rf_live_predictions_master(df, winner_classifier, home_pts_regressor, away_pts_regressor, 
                               total_pts_regressor,
                              home_spread_regressor, todays_date=datetime.today().strftime('%Y-%m-%d'),
                              feature_start_column="PREVIOUS_MATCHUP_RECORD")
```
>><span style="color:#000000">Function to consolidate all live Random Forest predictions into one dataframe, with the RF model predictions, market predictions, and game information, so it is more easily understood by the user, and so it can be used in live betting strategies. 

>><span style="color:#000000">Returns a master dataframe with all RandomForest predictions for each game: winner, home points, away points, total points, and points spread

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**winner_classifier** (RandomForestClassifier object). The model created from using historical data from the [backtesting](#backtesting) module and the [rf\_classify\_win\_loss](#rf_classify_win_loss) function. This model is used to predict the outcome of today's games.
* <span style="color:#000000">**home\_pts_regressor** (RandomForestRegressor object): RandomForestRegressor model as an object, returned when using the [rf\_predict\_home\_team\_points](#rf_predict_home_team_points) function in the [backtesting](#backtesting) module. This is used to make the actual predictions on today's games
* <span style="color:#000000">**away\_pts_regressor** (RandomForestRegressor object): RandomForestRegressor model as an object, returned when using the [rf\_predict\_away\_team\_points](#rf_predict_away_team_points) function in the [backtesting](#backtesting) module. This is used to make the actual predictions on today's games
* <span style="color:#000000">**total\_pts_regressor** (RandomForestRegressor object): RandomForestRegressor model as an object, returned when using the [rf\_predict\_total\_points](#rf_predict_total_points) function in the [backtesting](#backtesting) module. This is used to make the actual predictions on today's games
* <span style="color:#000000">**home\_spread_regressor** (RandomForestRegressor object): RandomForestRegressor model as an object, returned when using the [rf\_predict\_home\_spread](#rf_predict_home_spread) function in the [backtesting](#backtesting) module. This is used to make the actual predictions on today's games
* <span style="color:#000000">**todays_date** (string): Default: today's date, using the *datetime* package. Today's date, in the form of 'YYYY-MM-DD'. Leave this as default, unless you want to predict the next days game's at the end of the current day (rather than the default of predicting on the day of). Also consider manually changing if there are timezone differences.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.

>> ## <a name="live.predictors.SVM"><span style="color:#808080">SVM</span></a>

>>#### <a name = "svm_live_predict_winner"> <span style="color:#0052cc"> svm\_live\_predict\_winner </span></a> 

>>```Python
svm_live_predict_winner(df, classifier, todays_date=datetime.today().strftime('%Y-%m-%d'),
            feature_start_column="PREVIOUS_MATCHUP_RECORD")
```
>><span style="color:#000000">Function to simplify predicting the winner of a game that is yet to occur. Utilizes the *SVM* function, and the moving averages/cumulative average of each feature before the game to predict the game. Takes as input a cleaned and consolidated dataframe created from the [create\_final\_dataset](#create_final_dataset) or [multi\_season\_dataset](#multi_season_final_dataset) functions.
  
>><span style="color:#000000">Returns:  
 
>>* <span style="color:#000000">1 list: **[0] = y_pred** - an array with the win/loss prediction for each game, as a 1 or 0. 1 indicates the home team winning
* <span style="color:#000000">1 object: **[1] = todays_games** - the dataset including all features and (empty) results columns for the games happening today.

>><span style="color:#000000">Parameters:  
  
>>* <span style="color:#000000">**df** (pandas dataframe): a cleaned dataframe, created using the [builder](#dataCollection.builder) module.
* <span style="color:#000000">**classifier** (RandomForestClassifier object). The model created from using historical data from the [backtesting](#backtesting) module and the [svm\_classify\_win\_loss](#svm_classify_win_loss) function. This model is used to predict the outcome of today's games.
* <span style="color:#000000">**todays_date** (string): Default: today's date, using the *datetime* package. Today's date, in the form of 'YYYY-MM-DD'. Leave this as default, unless you want to predict the next days game's at the end of the current day (rather than the default of predicting on the day of). Also consider manually changing if there are timezone differences.
* <span style="color:#000000">**feature\_start_column** (string): Default = "PREVIOUS\_MATCHUP\_RECORD" (the column name created when using the [builder](#dataCollection.builder) module). The first column in the dataframe representing the first feature used in the training of the model. Every column after this model should represent a feature that was used in model training.
