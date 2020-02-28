import requests
import io
import pandas
import numpy
import dateutil.parser as dateparser


# Download a full season of games and return the raw input as a 
# pandas.dataframe
def _download_games(league, season):
    seasonstr = "{}{}".format(season, season + 1)
    url = "https://www.footballwebpages.co.uk/fixtures-results.csv?comp={}&season={}".format(league, seasonstr)
    response = requests.get(url)
    data = pandas.read_csv(io.StringIO(response.text))
    data['Season'] = season
    leaguename = {1:'Premiership', 2:'Championship', 3:'League One', 4:'League Two'}
    data['League'] = 'comp{}'.format(leaguename[league])
    return data[~data['Date'].isna()]

# Extract only the columns we need and parse dates
def _parse_games(rawgames):
    parsed = pandas.DataFrame()
    # These come from the original csv
    parsed['Date'] = rawgames['Date'].apply(lambda x: dateparser.parse(x, dayfirst=True).date())
    parsed['HomeTeam'] = rawgames['Home Team']
    parsed['AwayTeam'] = rawgames['Away Team']
    # These come from columns added in _download_games
    parsed['Season'] = rawgames['Season']
    parsed['League'] = rawgames['League']
    # Results
    parsed['HomePoints'] = numpy.NaN
    parsed['AwayPoints'] = numpy.NaN
    parsed['Result'] = numpy.NaN
    res = ~(rawgames['Status'] != 'FT')
    parsed.loc[res,'HomePoints'] = rawgames.loc[res,'H Score'].apply(int)
    parsed.loc[res,'AwayPoints'] = rawgames.loc[res,'A Score'].apply(int)
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
    if (league not in [1,2,3,4]):
        raise Exception("League should be an integer 1-4.")
    rawgames = _download_games(league, season)
    games = _parse_games(rawgames)
    return _filter_games(games, startdate, enddate)

