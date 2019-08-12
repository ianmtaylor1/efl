"""This module contains classes and functions for accessing game data."""

from . import orm

from sqlalchemy.orm import aliased
from sqlalchemy import and_

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
