# -*- coding: utf-8 -*-
"""
analysis.py

Contains useful functions for processing efl games that don't fit into any
particular class.
"""

import pandas
import numpy

###########################################################
### FUNCTIONS TO CREATE LEAGUE TABLES AND WIN MATRICES ####
###########################################################

def make_table(df):
    """Take in a data frame of games and transform it into a league table.
    Columns in the resulting table follow the format (H|A)(GF|GA|Pts) for
    (home/away) (goals for/goals against/points)."""
    # Pre-group for efficiency
    grp_home = df.groupby('hometeam', sort=True)
    grp_away = df.groupby('awayteam', sort=True)
    # Home Goals For, Home Goals Against, Away Goals For, Away Goals Against
    hgf = grp_home['homegoals'].sum()
    hga = grp_home['awaygoals'].sum()
    agf = grp_away['awaygoals'].sum()
    aga = grp_away['homegoals'].sum()
    # Home Wins, Home Draws, Home Wins, Home Draws
    hw = grp_home['result'].agg(lambda x: sum(x == 'H'))
    hd = grp_home['result'].agg(lambda x: sum(x == 'D'))
    aw = grp_away['result'].agg(lambda x: sum(x == 'A'))
    ad = grp_away['result'].agg(lambda x: sum(x == 'D'))
    # Build table
    table = pandas.DataFrame({
            'HGF':hgf, 'HGA':hga,
            'AGF':agf, 'AGA':aga,
            'HPts':3*hw + hd,
            'APts':3*aw + ad})
    return table

def make_matrix(df):
    """Take in a data frame of games and transform it into a matrix
    representing a directed multigraph of wins. E.g. if team 1 beat team 5 
    twice, then M[1,5] = 2. Draws are not included in the matrix."""
    teams = list(set(df['hometeam']) | set(df['awayteam']))
    teamidx = {t:i for i,t in enumerate(teams)}
    P = len(teams)
    mat = numpy.zeros(shape=[P,P])
    for i,row in df.iterrows():
        if row['result'] == 'H':
            mat[teamidx[row['hometeam']],teamidx[row['awayteam']]] += 1
        elif row['result'] == 'A':
            mat[teamidx[row['awayteam']],teamidx[row['hometeam']]] += 1
    return mat

