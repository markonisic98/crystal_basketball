# Crystal Basketball

Crystal Basketball is a project dedicated to predicting the outcome of an NBA game using machine learning techniques. While the program's main focus is to predict the winner of a specific game, it can also predict the point spread and total points of a game. Currently, this program is focused around the prediction of regular season games. 

The overarching goal of this project is to see if we can develop sophisticated models that can outperform bookmakers and produce a positive return betting in the market based on the models' – and eventually **one** consolidated model's – predictions. 

### The Data

Our program is currently capable of collecting both basic and advanced stats from game logs for for every game in any NBA season. Basic stats include your typical run-of-the-mill box stats: *shooting splits, points, rebounds, assists, blocks, steals, and turnovers*. Advanced stats include *true shooting %, offensive rating, and defensive rating*, among several other percentage/ratio based stats. These stats are scraped from [basketball-reference](https://www.basketball-reference.com). Full documentation for these stats can be found at the following links, by hovering over each of the stats columns (Using Toronto's 2021 season as an example):  
<https://www.basketball-reference.com/teams/TOR/2021/gamelog/>  
<https://www.basketball-reference.com/teams/TOR/2021/gamelog-advanced/>  


Obviously, we can't use the stats from a given game to predict its outcome. Otherwise the predictive model would have 100% prediction accuracy by simply looking at which team scored more points. This would also have no practical use. Instead, we take moving averages of these stats and use them as the features in the predictive model. The user can decide the different moving averages, and how many of them, they would like to collect. The cumulative season average of each stat before each game – exclusive of the game in question – will be collected by default.

More detailed information regarding the data collection functions and methodology can be found in the documentation for the **dataset_builder** and **dataset\_builder_helpers** files.


### Predictive Models
Currently, the model offers consolidated functions to predict game outcomes using a Random Forest model, and a Support Vector Machine (SVM) model. Users are more than welcome to utilize only the dataset building portion of this program to scrape data and utilize other predictive models. Soon, we hope to add several other ML techniques that may offer better predictive power.

For simplicity, the models are tasked with predicting the game outcomes from the perspective of the home team (i.e., whether or not home team will win and home team points spread) as opposed to naming the specific team names and their probability of winning. Though this can easily be derived by pulling the home and away team names for each game.

More information for the Machine Learning models used can be found in the documentation for the **random_forest\_predictions** and **svm_predictions** files.

### Betting Strategies
There are functions in the model that scrape historical odds for each game from *January 2015* onward. This scrapes the odds for each team to win in a specific game (gameline), over/under, and points spread.

There are currently two betting strategies that can be used for each predictive model/category combination to assess return on investment: simple and advanced. Though each model/category combo betting strategy differs slightly, a simple strategy generally uses a naive approach to betting, where it takes the model's prediction at face value and places a bet accordingly. An advanced strategy will take these predictions and only place a bet if a certain threshold difference between model prediction and market odds is reached. For example, this strategy would only recommend a gameline bet is placed if the implied odds of Team A winning (using model probabability of Team A winning) are 10% greater than the market odds suggest.

In the model/category combo mentioned above, the models have already been mentioned: Random Forest and SVM. The categories include: gameline betting, over/under betting, and points spread betting. 

More information for betting strategies can be found in the documentation for the **betting_strategies** file.

#### More Information

For more information, read through each module and function's section in the **documentation** file, and utilize the **examples** file for a more visual, and often quicker, introduction to the program.
