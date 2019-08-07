import requests
import io
import pandas
import dateutil.parser as dateparser


# Download a full season of games and return the raw input as a 
# pandas.dataframe
def _download_games(league, season):
    seasonstr = "{}{}".format(season % 100, (season + 1) % 100)
    leaguestr = {1:"E0", 2:"E1", 3:"E2", 4:"E3", 5:"EC"}[league]
    url = "http://www.football-data.co.uk/mmz4281/{}/{}.csv".format(seasonstr, leaguestr)
    response = requests.get(url)
    data = pandas.read_csv(io.StringIO(response.text))
    data['Season'] = season
    return data[~data['Date'].isna()]

# Extract only the columns we need and parse dates
def _parse_games(rawgames):
    parsed = pandas.DataFrame()
    parsed['Date'] = rawgames['Date'].apply(lambda x: dateparser.parse(x, dayfirst=True))
    parsed['HomeTeam'] = rawgames['HomeTeam']
    parsed['AwayTeam'] = rawgames['AwayTeam']
    parsed['HomePoints'] = rawgames['FTHG'].apply(int)
    parsed['AwayPoints'] = rawgames['FTAG'].apply(int)
    parsed['Result'] = rawgames['FTR']
    parsed['Season'] = rawgames['Season']
    parsed['League'] = rawgames['Div']
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
    if (league not in [1,2,3,4,5]):
        raise Exception("League should be an integer 1-5.")
    rawgames = _download_games(league, season)
    games = _parse_games(rawgames)
    return _filter_games(games, startdate, enddate)

