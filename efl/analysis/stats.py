#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py

Contains functions which compute statistics on sets of games (simulated or
real). Useful (mainly used) for EFLPredictor statistics.

Statistics for EFLPredictor are callables with the following (optional)
attributes:
    type_ - indicates the return type of the function. ('numeric', 'nominal',
        or 'ordinal'). Tells EFLPredictor how to summarize or plot this
        statistic. If not present, or None, then the stat cannot be summarized
        or plotted.
    precompute - function or iterable of functions that need to be
        precomputed before the wrapped function can be computed. The
        wrapped function should have one positional argument per element
        of precompute, and will receive the results of the precompute
        functions as positional arguments in the order they are listed. If
        None, or not present, a standard dataframe of game results will be
        passed in.
    name - The name to apply to this statistic for referency by the end user.
    sort - if type_ is 'nominal', a key function that is used to sort
        values. If None, Python's default ascending sort is used.

They can be created using this module in two ways: (1) by using the decorator
@stat, or (2) by subclassing BaseStat and implementing __call__.
"""

import functools
import numpy
import pandas


###########################################################
## BASE CLASS AND FUNCTION DECORATOR ######################
###########################################################


class BaseStat(object):
    """Base class for stat classes. Implements common features and provides
    a handy __init__ function."""
    
    def __init__(self, type_=None, precompute=None, name=None):
        if precompute is not None:
            try:
                self.precompute = tuple(precompute)
            except TypeError:
                self.precompute = (precompute,)
        self.type_ = type_
        self.name = name
    
    def _id(self):
        """Returns identifying attributes of the object for hashing and
        equality comparisons. Can be a single value or a tuple of values, but
        needs to be hashable. Only needs to differentiate between members of
        the same class - class comparison is automatically handled. Should be
        equal for two instances that would result in the same computed values
        of the statistic in __call__. It's okay to leave this unimplemented,
        but it may result in lots of unnecessary repeated computation in the
        EFLPredictor."""
        raise NotImplementedError()
    
    def __call__(self, x):
        """Computes the desired statistic on the data x."""
        raise NotImplementedError()
    
    def __hash__(self):
        try:
            return hash((type(self), self._id()))
        except NotImplementedError:
            return super().__hash__()
    
    def __eq__(self, other):
        if isinstance(other, BaseStat):
            try:
                return type(self) == type(other) and self._id() == other._id()
            except NotImplementedError:
                return super().__eq__(other)
        return NotImplemented


def stat(_func=None, *, type_=None, precompute=None, name=None, sort=None):
    """Decorator for creating stat functions for use in EFLPredictor. Uses the
    paradigm of a decorator that can be used with or without arguments. (See
    https://realpython.com/primer-on-python-decorators/). Also means that it
    can be used as a decorator, or as a wrapper to decorate an existing
    function. E.g. either of these works:
    >>> @stat(type_='nominal',name='My Cool Stat')
        def mystat(df):
            ...
    >>> mystat2 = stat(some_existing_function, type_='numeric', name='Bingo!')
    
    Parameters:
        type_ - return type of this function ("numeric", "ordinal", or 
            "nominal"). If none, the stat cannot be plotted or summarized.
        precompute - function or iterable of functions that need to be
            precomputed before the wrapped function can be computed. The
            wrapped function should have one positional argument per element
            of precompute, and will receive the results of the precompute
            functions as positional arguments in the order they are listed. If
            none, a standard dataframe of game results will be passed in.
        name - The name to apply to this statistic. If None, the function's
            __name__ attribute is used by EFLPredictor.
        sort - if type_ is 'nominal', a key function that is used to sort
            values. If None, Python's default ascending sort is used.
    """
    # Create the decorator
    def stat_decorator(func):
        @functools.wraps(func)
        # Wrap the function, passing all positional arguments
        def statwrapper(*args):
            return func(*args)
        # Create a tuple of precompute functions, attach to this function
        if precompute is not None:
            try:
                statwrapper.precompute = tuple(precompute)
            except TypeError:
                statwrapper.precompute = (precompute,)
        # Attach the type to this function
        if type_ is not None:
            statwrapper.type_ = type_
        # Attach the name to this function
        if name is not None:
            statwrapper.name = name
        # Attach the sort key to this function
        if sort is not None:
            statwrapper.sort = sort
        # Return the new wrapped function
        return statwrapper
    # Return the decorator, or the decorated function
    if _func is None:
        return stat_decorator
    else:
        return stat_decorator(_func)


###########################################################
## USEFUL PRECOMPUTE FUNCTIONS ############################
###########################################################


def table(df):
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
            'HGF':hgf, 'AGF':agf, 'GF':hgf + agf,
            'HGA':hga, 'AGA':aga, 'GA':hga + aga,
            'HGD':hgf - hga, 'AGD':agf - aga, 'GD': hgf + agf - hga - aga,
            'HPts':3*hw + hd, 'APts':3*aw + ad, 'Pts':3*hw + hd + 3*aw + ad})
    return table


@stat(precompute=table)
def rankings(t):
    """Take in a table and return a series indicating the rankings of teams
    according to that table. 1 = best, top of table, 2 = second best, etc."""
    t2 = pandas.DataFrame({'pts':t.loc[:,'Pts'],
                           'gd':t.loc[:,'GD'],
                           'gf':t.loc[:,'GF'],
                           'rnd':numpy.random.normal(size=len(t))})
    t2 = t2.sort_values(['pts','gd','gf','rnd'], ascending=False)
    t2['rank'] = range(1,len(t2)+1)
    return t2['rank']


def matrix(df):
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


###########################################################
## DERIVED STAT CLASSES AND FUNCTIONS #####################
###########################################################


class GameResult(BaseStat):
    """This class is a stat that returns, for a given game id, what the
    result is. (Home, Draw, Away)"""
    
    def __init__(self, gameid, games=None):
        # Determine the name
        if games is None:
            name = "Result {}".format(gameid)
        else:
            gamedf = games.to_dataframe(fit=True, predict=True)
            hometeam = gamedf.loc[gameid, 'hometeam']
            awayteam = gamedf.loc[gameid, 'awayteam']
            name = '{} vs {} Result'.format(hometeam, awayteam)
        # Construct the BaseStat
        super().__init__(type_='ordinal', name=name)
        # Save the game id
        self._gameid = gameid
    
    def _id(self):
        return self._gameid
    
    def __call__(self, df):
        return df.loc[self._gameid, 'result']
    
    @staticmethod
    def sort(v):
        # Sorts in the order H,D,A
        return {'H':1, 'D':2, 'A':3}.get(v, 4)


class GameScore(BaseStat):
    """This class is a stat that returns, for a given game id, what the
    score is as a tuple: (homegoals, awaygoals)"""
    
    def __init__(self, gameid, games=None):
        # Determine the name
        if games is None:
            name = "Score {}".format(gameid)
        else:
            gamedf = games.to_dataframe(fit=True, predict=True)
            hometeam = gamedf.loc[gameid, 'hometeam']
            awayteam = gamedf.loc[gameid, 'awayteam']
            name = '{} vs {} Score'.format(hometeam, awayteam)
        # Construct the BaseStat
        super().__init__(type_='ordinal', name=name)
        # Save the game id
        self._gameid = gameid
    
    def _id(self):
        return self._gameid
    
    def __call__(self, df):
        return (df.loc[self._gameid,'homegoals'], df.loc[self._gameid,'awaygoals'])
    
    @staticmethod
    def sort(v):
        # Sorts from largest home win to largest away win
        margin = v[0] - v[1]
        if margin >= 0:
            total = v[0] + v[1]
        else:
            total = -(v[0] + v[1])
        return (-margin, -total)


class GameMargin(BaseStat):
    """This class is a stat that returns, for a given game id, what the
    margin is: homegoals - awaygoals"""
    
    def __init__(self, gameid, games=None):
        # Determine the name
        if games is None:
            name = "Margin {}".format(gameid)
        else:
            gamedf = games.to_dataframe(fit=True, predict=True)
            hometeam = gamedf.loc[gameid, 'hometeam']
            awayteam = gamedf.loc[gameid, 'awayteam']
            name = '{} vs {} Margin'.format(hometeam, awayteam)
        # Construct the BaseStat
        super().__init__(type_='ordinal', name=name)
        # Save the game id
        self._gameid = gameid
    
    def _id(self):
        return self._gameid
    
    def __call__(self, df):
        return df.loc[self._gameid,'homegoals'] - df.loc[self._gameid,'awaygoals']
    
    @staticmethod
    def sort(v):
        # Sorts from largest home win to largest away win
        return -v


class GameTotalGoals(BaseStat):
    """This class is a stat that returns, for a given game id, what the
    total score is: homegoals + awaygoals"""
    
    def __init__(self, gameid, games=None):
        # Determine the name
        if games is None:
            name = "Total Goals {}".format(gameid)
        else:
            gamedf = games.to_dataframe(fit=True, predict=True)
            hometeam = gamedf.loc[gameid, 'hometeam']
            awayteam = gamedf.loc[gameid, 'awayteam']
            name = '{} vs {} Total Goals'.format(hometeam, awayteam)
        # Construct the BaseStat
        super().__init__(type_='ordinal', name=name)
        # Save the game id
        self._gameid = gameid
    
    def _id(self):
        return self._gameid
    
    def __call__(self, df):
        # This assumes there is only one row for every gameid
        return df.loc[self._gameid,'homegoals'] + df.loc[self._gameid,'awaygoals']


class LeaguePosition(BaseStat):
    """This class returns the team that is in the supplied position within 
    the league. (1 = Winner, 2 = runner-up, etc). Issues: Ties (points, goal
    difference, and goals for all equal) are arbitrarily decided. In real
    tables, teams share a rank."""
    
    def __init__(self, position):
        super().__init__(precompute=rankings, type_='nominal',
                         name="League Position {}".format(position))
        self._pos = position

    def _id(self):
        return self._pos
    
    def __call__(self, r):
        return r.sort_values(ascending=False).index[self._pos-1]


class TeamPosition(BaseStat):
    """This class returns the position within the league of the specified
    team. (1=Winner, 2=Runner-up, etc) Issues: Ties (points, goal
    difference, and goals for all equal) are arbitrarily decided. In real
    tables, teams share a rank."""
    
    def __init__(self, team):
        super().__init__(precompute=rankings, type_='ordinal', 
                         name="{} Position".format(team))
        self._team = team
        
    def _id(self):
        return self._team
    
    def __call__(self, r):
        return r.loc[self._team]


class TeamPoints(BaseStat):
    """This class is a stat that returns, for a given team, how many points
    the team has total (Points are 3 for a win, 1 for a draw.)"""
    
    def __init__(self, team):
        super().__init__(precompute=table, type_='numeric',
                         name='{} points'.format(team))
        self._team = team

    def _id(self):
        return self._team
    
    def __call__(self, t):
        return t.loc[self._team,'Pts']


@stat(type_='numeric')
def homewinpct(g):
    """Calculates the homefield advantage: what percentage of games were won
    by the home team."""
    return 100*sum(g['result'] == 'H')/g['result'].count()


@stat(type_='numeric')
def drawpct(g):
    """Calculates the likelihood of a draw: what percentage of games resulted
    in a draw."""
    return 100*sum(g['result'] == 'D')/g['result'].count()


@stat(type_='numeric', precompute=table)
def maxpoints(t):
    """Calculates the max points by a team. (Points are 3 for a win, 1 for a 
    draw.)"""
    return t['Pts'].max()


@stat(type_='numeric', precompute=table)
def minpoints(t):
    """Calculates the min points by a team. (Points are 3 for a win, 1 for a
    draw.)"""
    return t['Pts'].min()


@stat(type_='numeric', precompute=table)
def sdpoints(t):
    """Calculates the standard deviation of points by a team. (Points are 3
    for a win, 1 for a draw.)"""
    return t['Pts'].std()


@stat(type_='numeric', precompute=table)
def sdhomeadv(t):
    """Calculates the standard deviation of the homefield advantage for all
    teams. Homefield advantage is defined as home points minus away points."""
    return numpy.std(t['HPts'] - t['APts'])


@stat(type_='numeric', precompute=table)
def maxhomeadv(t):
    """Calculates the maximum homefield advantage for all teams. Homefield
    advantage is defined as home points minus away points."""
    return numpy.max(t['HPts'] - t['APts'])


@stat(type_='numeric', precompute=matrix)
def numrecip(M):
    """Calculates the number of reciprocal game pairs. (e.g. Team A beats
    Team B, then Team B beats Team A.)"""
    M2 = numpy.matmul(M, M)
    return M2.diagonal().sum() / 2


@stat(type_='numeric', precompute=matrix)
def numtriangles(M):
    """Calculates the number of 'triangles' (e.g. A > B > C > A)"""
    M3 = numpy.matmul(numpy.matmul(M, M), M)
    return M3.diagonal().sum() / 3


@stat(type_='numeric', name='Goals Index of Dispersion')
def goals_ind_disp(df):
    """Calculates the index of dispersion (var/mean) for all goals scored in
    all games."""
    allgoals = df['homegoals'].append(df['awaygoals'])
    return allgoals.var()/allgoals.mean()


@stat(type_='numeric', name='Goals Coefficient of Variation')
def goals_cv(df):
    """Calculates the coefficient of variation (sd/mean) for all goals scored
    in all games."""
    allgoals = df['homegoals'].append(df['awaygoals'])
    return allgoals.std()/allgoals.mean()


@stat(type_='numeric')
def avghomemargin(df):
    """Calculates the average home team margin of victory for all games."""
    return (df['homegoals']-df['awaygoals']).mean()

