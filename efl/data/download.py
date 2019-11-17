"""Functions intended to be used mainly by entry_points for processing data."""

from . import orm
from . import footballdata
from . import fixturedownload
from . import db

import sqlalchemy
import argparse
import difflib
import numpy

# Function to prompt for a missing team
def _prompt_missing_team(session, name, sourcename):
    print("No team entries for '{}'".format(name))
    team = None
    while team is None:
        choice = input("Enter a teamid, 'c' to create, 's' to show all, or 'b' to show best: ")
        if choice == "c":
            team = orm.Team(shortname=name)
        if choice == "s":
            for tm in session.query(orm.Team).order_by(orm.Team.shortname).all():
                print("{}: {}".format(tm.id, tm.shortname), end="")
                if (tm.longname is not None) and (len(tm.longname) > 0):
                    print(" ({})".format(tm.longname), end="")
                print()
        if choice == "b":
            tms = sorted(session.query(orm.Team).all(),
                         key=lambda x: difflib.SequenceMatcher(None, name, x.shortname).ratio(),
                         reverse=True)
            for tm in tms[:5]:
                print("{}: {}".format(tm.id, tm.shortname), end="")
                if (tm.longname is not None) and (len(tm.longname) > 0):
                    print(" ({})".format(tm.longname), end="")
                print()
        else:
            team = session.query(orm.Team).filter(orm.Team.id == choice).one_or_none()
    stn = orm.SourceTeamName(datasource=sourcename, name=name)
    stn.team = team
    return stn

def _prompt_missing_league(session, name, sourcename):
    print("No league entries for '{}'".format(name))
    league = None
    while league is None:
        choice = input("Enter a leagueid, 'c' to create, or 's' to show all: ")
        if choice == "c":
            shortname = input("Enter new league short name: ")
            longname = input("Enter new league long name: ")
            league = orm.League(shortname=shortname, longname=longname)
        if choice == "s":
            for lg in session.query(orm.League).order_by(orm.League.id).all():
                print("{}: {}".format(lg.id, lg.shortname), end="")
                if (lg.longname is not None) and (len(lg.longname) > 0):
                    print(" ({})".format(lg.longname), end="")
                print()
        else:
            league = session.query(orm.League).filter(orm.League.id == choice).one_or_none()
    sln = orm.SourceLeagueName(datasource=sourcename, name=name)
    sln.league = league
    return sln
    

def fetch_and_save(league, year, fetcher, sourcename):
    """Download and games for a given league and year, and save them to the
    EFL games database."""
    # Get the list of games from the provider
    games = fetcher(league, year)
    
    # Create session and tables
    orm.Base.metadata.create_all(db.connect())  # create tables
    session = db.Session()
    
    # SEASON ############################################################
    print("\nChecking for existence of season...")
    
    season = session.query(orm.Season).filter(orm.Season.start == year).one_or_none()
    if season is None:
        print("Creating new season ({}-{})".format(year, year+1))
        season = orm.Season(start=year, end=year+1, 
                            name="{}-{}".format(year, year+1))
        session.add(season)
        session.commit()

    # TEAM ###############################################################
    print("\nChecking for existence of teams...")
    
    # Check for team entries - create dict called "teams" to link team names
    # to orm.Team class instance.
    teamnames = list(set(games['HomeTeam']).union(set(games['AwayTeam'])))
    teams = {}
    for t in teamnames:
        stn = session.query(orm.SourceTeamName).filter(
                sqlalchemy.and_(orm.SourceTeamName.name == t, 
                                orm.SourceTeamName.datasource == sourcename)
                ).one_or_none()
        if stn is None:
            stn = _prompt_missing_team(session,t,sourcename)
            session.add(stn)
            session.commit()
        teams[t] = stn.team

    # LEAGUE ########################################################
    print("\nChecking for existence of leagues...")
    
    # Check for league entries - create dict called "leagues" to link league 
    # names to orm.League class instances
    leaguenames = list(set(games['League']))
    leagues = {}
    for l in leaguenames:
        sln = session.query(orm.SourceLeagueName).filter(
                sqlalchemy.and_(orm.SourceLeagueName.name == l,
                        orm.SourceLeagueName.datasource == sourcename)
                ).one_or_none()
        if sln is None:
            sln = _prompt_missing_league(session, l, sourcename)
            session.add(sln)
            session.commit()
        leagues[l] = sln.league
    
    # TEAM-LEAGUE ##################################################
    print("\nChecking for team/league memberships...")
    
    # Check for team-league entries for the given teams in the given season. 
    # Should match the current league
    hometeamleaguenames = set([(games.loc[i, "HomeTeam"], games.loc[i, "League"]) for i in games.index])
    awayteamleaguenames = set([(games.loc[i, "AwayTeam"], games.loc[i, "League"]) for i in games.index])
    teamleaguenames = list(hometeamleaguenames.union(awayteamleaguenames))
    for t,l in teamleaguenames:
        tl = session.query(orm.TeamLeague).filter(
                sqlalchemy.and_(orm.TeamLeague.teamid == teams[t].id,
                                orm.TeamLeague.seasonid == season.id)
                ).one_or_none()
        if tl is None:
            print("Adding '{}' to '{}' for season {}".format(teams[t].shortname, leagues[l].shortname, season.name))
            tl = orm.TeamLeague(teamid=teams[t].id, seasonid=season.id, leagueid=leagues[l].id)
            session.add(tl)
            session.commit()
        if tl.leagueid != leagues[l].id:
            raise Exception("Team '{}' in two leagues for season {}".format(teams[t].shortname, season.name))
    
    # GAMES ############################################################
    print("\nChecking for the existence of games...")
    
    for i in games.index:
        game = session.query(orm.Game).filter(
                sqlalchemy.and_(
                        orm.Game.date == games.loc[i,'Date'],
                        orm.Game.hometeamid == teams[games.loc[i,'HomeTeam']].id,
                        orm.Game.awayteamid == teams[games.loc[i,'AwayTeam']].id
                        )
                ).one_or_none()
        if game is None:
            print("Adding {} vs {}".format(teams[games.loc[i,"HomeTeam"]].shortname, 
                    teams[games.loc[i,"AwayTeam"]].shortname))
            game = orm.Game(date=games.loc[i,"Date"])
            game.hometeam = teams[games.loc[i,"HomeTeam"]]
            game.awayteam = teams[games.loc[i,"AwayTeam"]]
            game.season = season
            session.add(game)
        if (game.result is None) \
                and (not numpy.isnan(games.loc[i,"HomePoints"])) \
                and (not numpy.isnan(games.loc[i,"AwayPoints"])):
            print("Adding {}-{} result to {} vs {}".format(
                    games.loc[i,"HomePoints"], games.loc[i,"AwayPoints"],
                    game.hometeam.shortname, game.awayteam.shortname))
            # BUG FIX: int(...) below fixes weird error where integers get stored
            # in sqlite as BLOBs. Pandas problem, not reading in CSV as integer?
            # Idk it works now though.
            result = orm.GameResult(homegoals=int(games.loc[i,"HomePoints"]),
                                    awaygoals=int(games.loc[i,"AwayPoints"]),
                                    overtimes=0)
            game.result = result
    
    session.commit()
    session.close()


def console_download_games():
    """Function to be run as a console command entry point, initiates download
    of games and saves them to a sqlite database"""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Download and save EFL games.")
    parser.add_argument("-y", type=int, required=True,
            help="The season to download (e.g. 2018 for 18-19 season).")
    parser.add_argument("-l", type=int, choices=[1,2,3,4], required=True,
            help="The league to download: 1=Premier, 2=Championship, 3=League1, 4=League2")
    parser.add_argument("--config", "-c", type=str,
            help="Configuration file to override defaults.")
    # Parse args
    args = parser.parse_args()
    # Read the optionally supplied configuration file
    if args.config is not None:
        from .. import config
        config.parse(args.config)
    # Run the interactive data getting function
    if args.l in [1,2]:
        print("\nGetting data from fixturedownload.com")
        fetch_and_save(league=args.l, year=args.y, 
                       fetcher=fixturedownload.get_games, sourcename='fixturedownload')
    print("\nGetting data from footballdata.co.uk")
    fetch_and_save(league=args.l, year=args.y, 
                   fetcher=footballdata.get_games, sourcename='footballdata')


