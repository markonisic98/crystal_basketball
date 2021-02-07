# takes a cleaned dataframe with the predictions in the form that they are created by the 
# live predictions master function in the live.predictions module
import pandas as pd

def live_betting_simple_master(df, table_form=True, text_form=False, visual_form=False): 
    
    matchup_list=[None]*df.shape[0]
    i=0
    for home_team, away_team in zip(df['HOME_TEAM'], df['AWAY_TEAM']):
        matchup_list[i] = f'{home_team} vs. {away_team}'
        i+=1
        
    winner_bet=[None]*df.shape[0]
    spread_bet=[None]*df.shape[0]
    over_under_bet=[None]*df.shape[0]
    i=0
    for winner, spread_pred, OU_pred, market_spread, market_OU in zip(df['RF_WINNER_PRED'], 
                                    df['RF_HOME_SPREAD_PRED'], df['RF_TOTAL_PTS_PRED'],
                                    df['IMPLIED_HOME_SPREAD'], df['OVER_UNDER']):
        winner_bet[i]=winner
        if market_spread < 0:
            if spread_pred < market_spread:
                spread_bet[i] = "cover"
            else:
                spread_bet[i] = "against"
        else:
            if spread_pred < market_spread:
                spread_bet[i] = "against"
            else:
                spread_bet[i] = "cover"
        if OU_pred > market_OU:
            over_under_bet[i] = "over"
        else:
            over_under_bet[i] = "under"
        i+=1
    content = {'MATCHUP': matchup_list, 'WINNER': winner_bet, 'OVER/UNDER': over_under_bet, 'SPREAD': spread_bet}
    betting_master = pd.DataFrame(data=content)
    return betting_master

# currently only supports random forest models based on the name of the prediction columns being "RF_{prediction_type}"
# this can be easily changed later within the function to accept other column names
def live_betting_advanced_master(df, odds_threshold=0.2, over_under_threshold=0.05, spread_threshold=0.4,
                                table_form=True, text_form=False, visual_form=False):    
    
    matchup_list=[None]*df.shape[0]
    i=0
    for home_team, away_team in zip(df['HOME_TEAM'], df['AWAY_TEAM']):
        matchup_list[i] = f'{home_team} vs. {away_team}'
        i+=1
    
    
    winner_bet=[None]*df.shape[0]
    winner_bet_explanation=[None]*df.shape[0]
    spread_bet=[None]*df.shape[0]
    spread_bet_explanation=[None]*df.shape[0]
    over_under_bet=[None]*df.shape[0]
    over_under_bet_explanation=[None]*df.shape[0]
    i=0
    for (model_odds_home, model_odds_away, home_team, away_team, spread_pred, OU_pred, home_pts_pred, away_pts_pred, 
    market_odds_home, market_odds_away, market_spread, market_OU) in zip(df['RF_MODEL_ODDS_HOME_WIN'], 
                                df['RF_MODEL_ODDS_AWAY_WIN'], df['HOME_TEAM'], df['AWAY_TEAM'], df['RF_HOME_SPREAD_PRED'], 
                                df['RF_TOTAL_PTS_PRED'], df['RF_HOME_PTS_PRED'], df['RF_AWAY_PTS_PRED'], 
                                df['HOME_ODDS'], df['AWAY_ODDS'], df['IMPLIED_HOME_SPREAD'], df['OVER_UNDER']):
        
        # betting on winner of game (moneyline)
        if model_odds_home*(1+odds_threshold) < market_odds_home:
            winner_bet[i] = home_team 
            winner_bet_explanation[i] = (home_team + f' {round(((market_odds_home/model_odds_home) - 1)*100,0)}%' +
                                         ' more likely to win than market suggests.')
        elif model_odds_away*(1+odds_threshold) < market_odds_away:
            winner_bet[i] = away_team
            winner_bet_explanation[i] = (away_team + f' {round(((market_odds_away/model_odds_away) - 1)*100,0)}%' +
                                         ' more likely to win than market suggests.')
        else:
            if model_odds_home < market_odds_home:
                winner_bet[i] = "Don't bet"
                winner_bet_explanation[i] = (home_team + f' only {round(((market_odds_home/model_odds_home) - 1)*100,0)}%' +
                    ' more likely to win than market suggests.')
            else: 
                winner_bet[i] = "Don't bet"
                winner_bet_explanation[i] = (away_team + f' only {round(((market_odds_away/model_odds_away) - 1)*100,0)}%' +
                    ' more likely to win than market suggests.')
        
        # betting on over/under
        if OU_pred*(1+over_under_threshold) < market_OU and (home_pts_pred + away_pts_pred) < market_OU:
            over_under_bet[i] = "Under"
            over_under_bet_explanation[i] = f'Under {round(abs(((market_OU/OU_pred)-1)*100),0)}% lower hits a majority of the time.'
        elif OU_pred > market_OU*(1+over_under_threshold) and (home_pts_pred + away_pts_pred) > market_OU:
            over_under_bet[i] = "Over"
            over_under_bet_explanation[i] = f'Over {round(abs(((market_OU/OU_pred)-1)*100),0)}% higher hits a majority of the time.'
        else:
            over_under_bet[i] = "Don't bet"
            if OU_pred*(1+over_under_threshold) < market_OU and (home_pts_pred + away_pts_pred) >= market_OU:
                over_under_bet_explanation[i] = "Individual home/away pts predictions did not line up with total pts prediction"
            elif OU_pred > market_OU*(1+over_under_threshold) and (home_pts_pred + away_pts_pred) <= market_OU:
                over_under_bet_explanation[i] = "Individual home/away pts predictions did not line up with total pts prediction"
            else:
                if OU_pred > market_OU:
                    over_under_bet_explanation[i] = (f"Over/Under prediction only {round(abs(((market_OU/OU_pred)-1)*100),0)}% " + 
                                                         "higher than market Over/Under")
                else:
                    over_under_bet_explanation[i] = (f"Over/Under prediction only {round(abs(((market_OU/OU_pred)-1)*100),0)}% " + 
                                                         "lower than market Over/Under")
        
        # betting on spread
        # if one spread is positive and the other negative, in all cases it means that it passes the difference threshold
        if market_spread > 0:
            if spread_pred*(1+spread_threshold) < market_spread and (home_pts_pred-away_pts_pred) < market_spread:
                spread_bet[i] = "Against"
                # account for posi vs. neg values in the market vs. model spread when applying percentage diff
                if spread_pred < 0:
                    spread_bet_explanation[i] = (f'spread guesses the opposite team to the market favorite to win.')
                else:
                    spread_bet_explanation[i] = (f'spread {abs(round(((market_spread/spread_pred)-1)*100,0))}% more likely to go ATS ' +
                    'than market suggests')
            elif spread_pred > market_spread*(1+spread_threshold) and (home_pts_pred-away_pts_pred) > market_spread:
                spread_bet[i] = "Cover"
                # account for posi vs. neg values in the market vs. model spread when applying percentage diff
                if spread_pred > 0:
                    spread_bet_explanation[i] = (f'spread guesses the opposite team to the market favorite to win.')
                else:
                    spread_bet_explanation[i] = (f'spread {abs(round(((market_spread/spread_pred)-1)*100,0))}% more likely to cover ' +
                    'than market suggests')
            else:
                spread_bet[i] = "Don't bet"
                if spread_pred*(1+spread_threshold) < market_spread and (home_pts_pred-away_pts_pred) > market_spread:
                    spread_bet_explanation[i] = "Individual home/away points predictions did not line up with spread prediction."
                elif spread_pred > market_spread*(1+spread_threshold) and (home_pts_pred-away_pts_pred) < market_spread:
                    spread_bet_explanation[i] = "Individual home/away points predictions did not line up with spread prediction."
                else:
                    spread_bet_explanation[i] = f'spread prediction within {abs(spread_threshold*100)}% of market spread.'
        else:
            if spread_pred*(1+spread_threshold) > market_spread and (home_pts_pred-away_pts_pred) > market_spread:
                spread_bet[i] = "Against"
                # account for posi vs. neg values in the market vs. model spread when applying percentage diff
                if spread_pred > 0:
                    spread_bet_explanation[i] = (f'spread guesses the opposite team to the market favorite to win.')
                else:
                    spread_bet_explanation[i] = (f'spread {abs(round(((market_spread/spread_pred)-1)*100,0))}% more likely to go ATS ' +
                    'than market suggests')
            elif spread_pred < market_spread*(1+spread_threshold) and (home_pts_pred-away_pts_pred) < market_spread:
                spread_bet[i] = "Cover"
                # account for posi vs. neg values in the market vs. model spread when applying percentage diff
                if spread_pred > 0:
                    spread_bet_explanation[i] = (f'spread guesses the opposite team to the market favorite to win.')
                else:
                    spread_bet_explanation[i] = (f'spread {abs(round(((market_spread/spread_pred)-1)*100,0))}% more likely to cover ' +
                    'than market suggests')
            else:
                spread_bet[i] = "Don't bet"
                if spread_pred*(1+spread_threshold) > market_spread and (home_pts_pred-away_pts_pred) < market_spread:
                    spread_bet_explanation[i] = "Individual home/away points predictions did not line up with spread prediction."
                elif spread_pred < market_spread*(1+spread_threshold) and (home_pts_pred-away_pts_pred) > market_spread:
                    spread_bet_explanation[i] = "Individual home/away points predictions did not line up with spread prediction."
                else:
                    spread_bet_explanation[i] = f'spread prediction within {abs(spread_threshold*100)}% of market spread.'
        
        i+=1
        
    content = {'MATCHUP': matchup_list, 'WINNER': winner_bet, 'WINNER_EXPLANATION': winner_bet_explanation, 
               'OVER/UNDER': over_under_bet, 'OVER_UNDER_EXPLANATION': over_under_bet_explanation, 'SPREAD': spread_bet, 'SPREAD_EXPLANATION': spread_bet_explanation}
    betting_master = pd.DataFrame(data=content)
    return betting_master

