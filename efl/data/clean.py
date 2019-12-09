# -*- coding: utf-8 -*-
"""
clean.py - code to do database integrity checks, mainly used from command line
"""

from . import orm
from . import db
import datetime
import itertools

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

def clean_db():
    """Run several database cleaning operations."""
    session = db.Session()
    
    # Within a league, within a season, search for duplicate home/away pairs
    print("\n** Potential duplicate matches...")
    seasons = session.query(orm.Season).order_by(orm.Season.start.desc())
    for s in seasons:
        print("* {}-{}".format(s.start, s.end))
        games = session.query(orm.Game)\
                       .filter(orm.Game.seasonid == s.id)\
                       .order_by(orm.Game.date)
        for (g1,g2) in itertools.combinations(games, 2):
            if (g1.hometeamid == g2.hometeamid) and (g1.awayteamid == g2.awayteamid):
                _display_game(g1)
                _display_game(g2)
    
    # Past games without results
    today = datetime.date.today()
    pastgames = session.query(orm.Game).filter(orm.Game.date < today).all()
    print("\n** Past games without results...")
    for g in pastgames:
        if g.result is None:
            _display_game(g)
    
    # Prompt for deleting games 
    print("\n** Delete games...")
    while _prompt_delete(session):
        pass
    
    # Commit any (for now nonexistent) changes
    session.commit()