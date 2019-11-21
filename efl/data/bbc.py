import requests
import pandas
import dateutil.parser as dateparser
import bs4


def _download_page(league, year, month, results=False):
    """Downloads a page from the bb.co.uk website and returns its html.
    Parameters:
        league: a number 1-4, 1 for Premier League, to 4 for League 2.
        year, month: the month to fetch matches for
        results: Only has effect when fetching the current month. By default,
            future fuxtures are downloaded. But if results=True, past results
            for the current month are fetched.
    """
    leaguestr = {1:"premier-league", 2:"championship", 
                 3:"league-one", 4:"league-two"}[league]
    url = 'https://www.bbc.com/sport/football/{}/scores-fixtures/{}-{:02d}'.format(leaguestr, year, month)
    if results:
        url += '?filter=results'
    response = requests.get(url)
    return response.text
    
def _parse_page(pagetext):
    """Parses the html of a fetched page into a pandas.DataFrame of games."""
    soup = bs4.BeautifulSoup(pagetext, features='html.parser')
    match_blocks = soup.find_all('div', attrs={'class':'qa-match-block'})
    return pandas.concat((_parse_block(m) for m in match_blocks), ignore_index=True)

def _parse_block(block):
    date = dateparser.parse(block.contents[0].text)
    items = block.find_all('li')
    matches = pandas.concat((_parse_item(i) for i in items), ignore_index=True)
    matches['Date'] = date
    return matches

def _parse_item(item):
    return pandas.DataFrame({'HomeTeam':['Home'],
                             'AwayTeam':['Away'],
                             'HomePoints':[0],
                             'AwayPoints':[0],
                             'Result':['D']})