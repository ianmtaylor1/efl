# -*- coding: utf-8 -*-
"""
clean.py - code to do database integrity checks, mainly used from command line
"""

from . import orm
from . import db
import datetime
import itertools
import sqlalchemy
import argparse

def _display_game(g):
    if g.result is None:
        resultstr = ""
    else:
        resultstr = "Result: {}-{}".format(g.result.homegoals,
                                           g.result.awaygoals)
    print("{}: {} {} vs {} {}".format(g.id, g.date.strftime("%Y-%m-%d"), 
                  g.hometeam.shortname, g.awayteam.shortname,
                  resultstr))

def _prompt_delete(sess):
    gid = input("Enter game id to delete, or leave blank to quit: ")
    game = sess.query(orm.Game).filter(orm.Game.id == gid).one_or_none()
    if game is not None:
        print("Deleting ",end="")
        _display_game(game)
        if game.result is not None:
            sess.delete(game.result)
        sess.delete(game)
        return True
    return False

def _check_season(session, year):
    """Run several database cleaning operations for a given season."""
    problems = False
    
    season = session.query(orm.Season).filter(orm.Season.start == year).one_or_none()
    
    print("\n\n** {}-{}".format(season.start, season.end))
    
    # Within a league, within a season, search for duplicate home/away pairs
    print("\n* Potential duplicate matches...")
    games = session.query(orm.Game)\
                   .filter(orm.Game.seasonid == season.id)\
                   .order_by(orm.Game.date)
    for (g1,g2) in itertools.combinations(games, 2):
        if (g1.hometeamid == g2.hometeamid) and (g1.awayteamid == g2.awayteamid):
            problems = True
            _display_game(g1)
            _display_game(g2)
    
    # Past games without results
    today = datetime.date.today()
    pastgames = session.query(orm.Game).outerjoin(orm.GameResult).filter(
            sqlalchemy.and_(orm.Game.date < today,
                            orm.GameResult.id == None)
            ).all()
    print("\n* Past games without results...")
    for g in pastgames:
        problems = True
        _display_game(g)
    
    # Prompt for deleting games 
    if problems:
        print("\n* Delete games...")
        while _prompt_delete(session):
            pass
    
    # How many games did a team play in a season?
    print("\n* Incorrect game counts...")
    leagues = session.query(orm.League)
    for lg in leagues:
        print("{}:".format(lg.shortname))
        teams = session.query(orm.Team).join(orm.TeamLeague).filter(
                sqlalchemy.and_(orm.TeamLeague.leagueid == lg.id,
                                orm.TeamLeague.seasonid == season.id)
                ).all()
        expgames = len(teams) - 1
        for t in teams:
            homecount = session.query(orm.Game).filter(
                    sqlalchemy.and_(orm.Game.seasonid == season.id,
                                    orm.Game.hometeamid == t.id)
                    ).count()
            awaycount = session.query(orm.Game).filter(
                    sqlalchemy.and_(orm.Game.seasonid == season.id,
                                    orm.Game.awayteamid == t.id)
                    ).count()
            if (homecount != expgames) or (awaycount != expgames):
                print("  {} {}: Home={}, Away={}. (Expected={})".format(
                        t.id, t.shortname, homecount, awaycount, expgames))


def clean_db():
    """Function to be run as a console command entry point, initiates cleaning
    of efl games db."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Download and save EFL games.")
    parser.add_argument("-y", type=int, required=False,
            help="The season to check (e.g. 2018 for 18-19 season).")
    parser.add_argument("--config", "-c", type=str,
            help="Configuration file to override defaults.")
    # Parse args
    args = parser.parse_args()
    # Read the optionally supplied configuration file
    if args.config is not None:
        from .. import config
        config.parse(args.config)
    
    session = db.Session()
    
    # If a year was provided, check that year. Otherwise check all years
    if args.y is not None:
        _check_season(session, args.y)
    else:
        years = [s.start for s in session.query(orm.Season).order_by(orm.Season.start).all()]
        for y in years:
            _check_season(session, y)
    
    session.commit()