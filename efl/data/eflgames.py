"""This module contains classes and functions for accessing game data."""

from . import orm

from sqlalchemy.orm import aliased
from sqlalchemy import and_
import pandas
import numpy

class EFLGames(object):
    """Class representing a read-only view of a subset of EFL games, for 
    model building.
    
    Each instance has three objects:
        fit - games to be used for fitting the model. Should all have results.
            (Results will also be predicted from the posterior for these games.)
        predict - games not used for fitting, but for which we want
            predictions of the result.
        teams - A list of teams to be accounted for in the model. May contain
            more teams than are actually represented in the games."""
    
    def __init__(self, games_fit, games_predict, teams):
        self.fit = games_fit
        self.predict = games_predict
        self.teams = teams
    
    # Class methods for creating instances
    
    @classmethod
    def from_season(cls, dbsession, seasonid, leagueid, asof_date=None): 
        """Build a data set from a given season and league.
        asof_date allows for a date to be set, after which games are assumed to
        not have results. (Good for running models historically.)"""
        # Build the team query
        teamquery = dbsession.query(orm.TeamLeague)\
                .filter(orm.TeamLeague.seasonid == seasonid)\
                .filter(orm.TeamLeague.leagueid == leagueid)
        # Build the game query
        htl = aliased(orm.TeamLeague)
        atl = aliased(orm.TeamLeague)
        gamequery = dbsession.query(orm.Game)\
                .join(htl, and_(htl.teamid == orm.Game.hometeamid,
                                htl.seasonid == orm.Game.seasonid))\
                .join(atl, and_(atl.teamid == orm.Game.awayteamid,
                                atl.seasonid == orm.Game.seasonid))\
                .filter(htl.leagueid == leagueid)\
                .filter(atl.leagueid == leagueid)\
                .filter(orm.Game.seasonid == seasonid)
        # Get the data
        games = gamequery.all()
        teams = [tl.team for tl in teamquery.all()]
        if asof_date is None:
            asof_date = max(g.date for g in games if g.result is not None)
        # Create and return the object
        games_fit = [g for g in games if (g.date <= asof_date) and (g.result is not None)]
        games_predict = [g for g in games if (g.date > asof_date) or (g.result is None)]
        return cls(games_fit, games_predict, teams)
    
    @classmethod
    def from_ids(cls, dbsession, fit_ids=[], predict_ids=[]):
        """Build a dataset from a list of game ids."""
        # Fit list
        if len(fit_ids) > 0:
            fitgames = dbsession.query(orm.Game)\
                    .filter(orm.Game.id in fit_ids)\
                    .all()
        else:
            fitgames = []
        # Predict list
        if len(predict_ids) > 0:
            predictgames = dbsession.query(orm.Game)\
                    .filter(orm.Game.id in predict_ids)\
                    .all()
        else:
            predictgames = []
        # Take all teams from all games
        teams = list(set(g.hometeam for g in fitgames) 
                     | set(g.awayteam for g in fitgames)
                     | set(g.hometeam for g in predictgames)
                     | set(g.awayteam for g in predictgames))
        # Build the object and return it
        return cls(fitgames, predictgames, teams)
    
    # Instance methods
    
    def to_dataframe(self, fit=True, predict=False):
        """Returns these games in a pandas.DataFrame with the gameid as the
        index, and the following columns (in order):
            date - date the game took place
            hometeam - unique short name of home team
            awayteam - unique short name of away team
            homegoals - home goals, if available (NA otherwise)
            awaygoals - away goals, if available (NA otherwise)
            result - match result ('H','A','D') if available (NA otherwise)
        If fit is true, the games in 'fit' will be included. If predict is
        true, the games in 'predict' will be included with results, if
        available.
        """
        if (not (fit or predict)):
            raise Exception("Either fit or predict or both must be true")
        if fit:
            # We can assume all games in fit have results
            fitdf = pandas.DataFrame({
                    'gameid':    [g.id for g in self.fit],
                    'date':      [g.date for g in self.fit],
                    'hometeam':  [g.hometeam.shortname for g in self.fit],
                    'awayteam':  [g.awayteam.shortname for g in self.fit],
                    'homegoals': [g.result.homegoals for g in self.fit],
                    'awaygoals': [g.result.awaygoals for g in self.fit]
                    })
        else:
            fitdf = None
        if predict:
            # Function to convert a game to a dictionary of items for the df
            def gametodict(g):
                d = {'gameid':g.id,
                     'date':g.date,
                     'hometeam':g.hometeam.shortname,
                     'awayteam':g.awayteam.shortname}
                if g.result is None:
                    d['homegoals'] = numpy.NaN
                    d['awaygoals'] = numpy.NaN
                else:
                    d['homegoals'] = g.result.homegoals
                    d['awaygoals'] = g.result.awaygoals
                return d
            predictdf = pandas.DataFrame([gametodict(g) for g in self.predict])
        else:
            predictdf = None
        # Combine and fill in result
        df = pandas.concat([fitdf,predictdf])
        df['result'] = numpy.NaN
        df.loc[df['homegoals'] < df['awaygoals'], 'result'] = 'A'
        df.loc[df['homegoals'] > df['awaygoals'], 'result'] = 'H'
        df.loc[df['homegoals'] == df['awaygoals'], 'result'] = 'D'
        return df.set_index('gameid')

def seasonid(session, start_year):
    """Return a unique seasonid from the database based on the season's start
    year."""
    season = session.query(orm.Season)\
            .filter(orm.Season.start == start_year)\
            .one_or_none()
    if season is None:
        return None
    else:
        return season.id

def leagueid(session, short_name):
    """Return a unique leagueid from the database based on the league's short
    name."""
    league = session.query(orm.League)\
            .filter(orm.League.shortname == short_name)\
            .one_or_none()
    if league is None:
        return None
    else:
        return league.id 
    
def teamid(session, short_name):
    """Return a unique teamid from the database based on the team's short
    name."""
    team = session.query(orm.Team)\
            .filter(orm.Team.shortname == short_name)\
            .one_or_none()
    if team is None:
        return None
    else:
        return team.id
