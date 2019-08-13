"""Module containing the object relational model for the EFL games database."""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.types import Integer, String, Date, Boolean
from sqlalchemy.orm import relationship


Base = declarative_base()


class Team(Base):
    __tablename__ = 'team'
    
    id = Column(Integer, primary_key=True)
    shortname = Column(String, nullable=False, unique=True)
    longname = Column(String, unique=True)
    location = Column(String)
    
    homegames = relationship("Game", foreign_keys="Game.hometeamid", back_populates="hometeam")
    awaygames = relationship("Game", foreign_keys="Game.awayteamid", back_populates="awayteam")
    
    def __repr__(self):
        return "<Team(shortname='{}', longname='{}', location='{}')>".format(
                self.shortname, self.longname, self.location)


class League(Base):
    __tablename__ = 'league'
    
    id = Column(Integer, primary_key=True)
    shortname = Column(String, nullable=False, unique=True)
    longname = Column(String, unique=True)
    
    def __repr__(self):
        return "<Division(shortname='{}', longname='{}')>".format(
                self.shortname, self.longname)


class Season(Base):
    __tablename__ = 'season'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    start = Column(Integer, nullable=False)
    end = Column(Integer)
    
    games = relationship("Game", back_populates="season")
    
    def __repr__(self):
        return "<Season(name='{}', start='{}', end='{}')>".format(
                self.name, self.start, self.end)


class Game(Base):
    __tablename__ = 'game'
    __table_args__ = (
            UniqueConstraint('date', 'hometeamid', 'awayteamid'),
            )
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    seasonid = Column(Integer, ForeignKey('season.id'))
    hometeamid = Column(Integer, ForeignKey('team.id'), nullable=False)
    awayteamid = Column(Integer, ForeignKey('team.id'), nullable=False)
    neutralsite = Column(Boolean, nullable=False, default=False)
    comments = Column(String)
    
    season = relationship("Season", back_populates="games")
    hometeam = relationship("Team", foreign_keys=[hometeamid], back_populates="homegames")
    awayteam = relationship("Team", foreign_keys=[awayteamid], back_populates="awaygames")
    result = relationship("GameResult", back_populates="game", uselist=False)
    
    def __repr__(self):
        return "<Game(date='{}', seasonid='{}', hometeamid='{}', awayteamid='{}', neutralsite='{}', comments='{}')>".format(
                self.date, self.seasonid, self.hometeamid, self.awayteamid, self.neutralsite, self.comments)


class GameResult(Base):
    __tablename__ = 'gameresult'
    
    id = Column(Integer, ForeignKey('game.id'), primary_key=True)
    homegoals = Column(Integer, CheckConstraint('homegoals >= 0'), nullable=False)
    awaygoals = Column(Integer, CheckConstraint('awaygoals >= 0'), nullable=False)
    overtimes = Column(Integer, CheckConstraint('overtimes >= 0'))
    comments = Column(String)
    
    game = relationship("Game", back_populates="result")
    
    def __repr__(self):
        return "<GameResult(homegoals='{}', awaygoals='{}', overtimes='{}', comments='{}')>".format(
                self.homegoals, self.awaygoals, self.overtimes, self.comments)


class TeamLeague(Base):
    __tablename__ = 'teamleague'
    
    teamid = Column(Integer, ForeignKey("team.id"), primary_key=True, nullable=False)
    leagueid = Column(Integer, ForeignKey("league.id"), nullable=False)
    seasonid = Column(Integer, ForeignKey("season.id"), primary_key=True, nullable=False)
    
    team = relationship("Team")
    league = relationship("League")
    season = relationship("Season")
    

class SourceTeamName(Base):
    __tablename__ = 'sourceteamname'
    
    datasource = Column(String, primary_key=True, nullable=False)
    teamid = Column(Integer, ForeignKey("team.id"), nullable=False)
    name = Column(String, primary_key=True, nullable=False)
    
    team = relationship("Team")


class SourceLeagueName(Base):
    __tablename__ = 'sourceleaguename'
    
    datasource = Column(String, primary_key=True, nullable=False)
    leagueid = Column(Integer, ForeignKey("league.id"),  nullable=False)
    name = Column(String, primary_key=True, nullable=False)
    
    league = relationship("League")
