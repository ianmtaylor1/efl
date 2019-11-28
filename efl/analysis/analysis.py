# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:46:23 2019

@author: Ian Taylor
"""

def add_info(df, games):
    """Take a data frame including a column called 'gameid', and add info
    from an EFLGames object (date, home team, away team)."""
    games_df = games.to_dataframe(fit=True, predict=True)
    return df.merge(games_df[['gameid','date','hometeam','awayteam']],
                    on = 'gameid', validate = 'm:1')