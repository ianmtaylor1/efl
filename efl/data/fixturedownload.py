import requests
import io
import pandas
import numpy
import dateutil.parser as dateparser

# Download a full season of games and return the raw download as a 
# pandas.dataframe
def _download_games(league, season):
    leaguestr = {1:"epl", 2:"championship"}[league]
    url = "https://fixturedownload.com/download/{}-{}-GMTStandardTime.csv".format(leaguestr, season)
    response = requests.get(url)
    data = pandas.read_csv(io.StringIO(response.text))
    data['Season'] = season
    data['Div'] = leaguestr
    return data[~data['Date'].isna()]

# Extract only the columns we need and parse dates
def _parse_games(rawgames):
    parsed = pandas.DataFrame()
    parsed['Date'] = rawgames['Date'].apply(lambda x: dateparser.parse(x, dayfirst=True).date())
    parsed['HomeTeam'] = rawgames['Home Team']
    parsed['AwayTeam'] = rawgames['Away Team']
    parsed['Season'] = rawgames['Season']
    parsed['League'] = rawgames['Div']
    # Results
    parsed['HomePoints'] = numpy.NaN
    parsed['AwayPoints'] = numpy.NaN
    parsed['Result'] = numpy.NaN
    res = ~rawgames['Result'].isna()
    parsed.loc[res,'HomePoints'] = rawgames.loc[res,'Result'].apply(lambda x: int(x.split('-')[0]))
    parsed.loc[res,'AwayPoints'] = rawgames.loc[res,'Result'].apply(lambda x: int(x.split('-')[1]))
    parsed.loc[parsed['HomePoints']>parsed['AwayPoints'],'Result'] = 'H'
    parsed.loc[parsed['HomePoints']<parsed['AwayPoints'],'Result'] = 'A'
    parsed.loc[parsed['HomePoints']==parsed['AwayPoints'],'Result'] = 'D'
    return parsed

# Filter to just between startdate and enddate, inclusive
def _filter_games(games, startdate, enddate):
    if startdate is None:
        startdate = min(games['Date'])
    if enddate is None:
        enddate = max(games['Date'])
    return games[(games['Date'] >= startdate) & (games['Date'] <= enddate)]

def get_games(league, season, startdate=None, enddate=None):
    """Returns a pandas DataFrame of all games taking place in that division,
    during that season, between startdate and enddate.
    
    league - an integer. 1 for Premier League, to 5 for conference.
    season - an integer. The year in which the season started.
    startdate, enddate - optional dates. If not specified, all games
        from the season will be returned.
    """
    if (league not in [1,2]):
        raise Exception("League should be an integer 1-2.")
    rawgames = _download_games(league, season)
    games = _parse_games(rawgames)
    return _filter_games(games, startdate, enddate)