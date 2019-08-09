"""This module contains classes and functions for accessing game data."""

# TODO: SESSION SCOPE!!

from . import engine
from . import orm

class EFLGames(object):
    """Class representing a read-only view of a subset of EFL games."""

    def __init__(self, seasonid=None, leagueid=None, 
            startdate=None, enddate=None):
        """Initialize the object. Any supplied filters are applied jointly,
        with 'and'. startdate and enddate are inclusive."""
        self.session = orm.Session(bind=engine.connect())
        # Build the game query
        gamequery = self.session.query(orm.Game)
        if (seasonid is not None) or (leagueid is not None):
            print("WARNING: League and season filter not implemented.")
        if startdate is not None:
            gamequery = gamequery.filter(orm.Game.date >= startdate)
        if enddate is not None:
            gamequery = gamequery.filter(orm.Game.date <= enddate)
        # Get the data
        self.games = gamequery.all()
        self.teams = list(set(
                [g.hometeam for g in self.games]
                + [g.awayteam for g in self.games]
                ))
    
    def addteam(self, teamid, exist_ok=True):
        """Adds a team to this object. Useful for when a team is in the league
        but hasn't played any games yet in the season. If exist_ok is false, 
        this method will raise an exception."""
        session = orm.Session(bind=engine.connect())
        team = session.query(orm.Team)\
                .filter(orm.Team.id == teamid)\
                .one_or_none()
        if team is None:
            raise Exception('Team ID {} does not exist.'.format(teamid))
        if (not exist_ok) and team in self.teams:
            raise Exception('Team ID {} already in data.'.format(teamid))
        self.teams = list(set(self.teams + [team]))


def seasonid(start_year):
    """Return a unique seasonid from the database based on the season's start
    year."""
    session = orm.Session(bind=engine.connect())
    season = session.query(orm.Season)\
            .filter(orm.Season.start == start_year)\
            .one_or_none()
    if season is None:
        return(None)
    else:
        return(season.id)

def leagueid(short_name):
    """Return a unique leagueid from the database based on the league's short
    name."""
    session = orm.Session(bind=engine.connect())
    league = session.query(orm.League)\
            .filter(orm.League.shortname == short_name)\
            .one_or_none()
    if league is None:
        return(None)
    else:
        return(league.id)
    
def teamid(short_name):
    """Return a unique teamid from the database based on the team's short
    name."""
    session = orm.Session(bind=engine.connect())
    team = session.query(orm.Team)\
            .filter(orm.Team.shortname == short_name)\
            .one_or_none()
    if team is None:
        return(None)
    else:
        return(team.id)
