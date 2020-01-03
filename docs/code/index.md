---
title: Code Overview
layout: main
---

# Package Overview

This project is built as the `efl` package to be installed by pip. This package has several main components, each designed to do a different task. Below is the outline of each component.

## data
Code in this package is responsible for storing and retrieving data about EFL matches from the database. It is based on [sqlalchemy](https://www.sqlalchemy.org/). The first step in using the `efl` package is usually getting data from this package.

## model
Code in this package contains Python and [Stan](https://mc-stan.org/) code for the models themselves. This is the workhorse of the program. Each model has its own Python class, all of which inherit from a base class. Therefore many functions of a model are common to all models.

Models are built by passing an `EFLGames` object to the appropriate class, then waiting for the Stan code to run. Once complete, the model object can be used to summarize its internal parameters, make plots of posterior samples, or predict the outcomes of observed or unobserved games.

## analysis
Code in this package is responsible for taking output from a fitted model object and analyzing its predictions about future games. It also contains the code for performing posterior predictive checks on match results and scores predicted by a model.

## config
Code in this package sets global configuration options - such as where the local database is saved, where compiled Stan models are cached, etc.


# Console Commands
The `efl` package also installs console commands to be run from the command line.

## `download-efl-games`
This command queries online data sources and fetches new results. It then updates the local database with the new info, prompting the user as necessary.

It uses [football-data.co.uk](http://www.football-data.co.uk/) for past results in the Premier League, Championship, League One, and League Two. It uses [fixturedownload.com](https://fixturedownload.com/) for future games in the Premier League and Championship. No future games are currently fetched for League One or League Two.

## `clean-efl-db`
This command is intended to be run after downloading new games. It searches for potential duplicates, games which may be missing results, and teams that play more or fewer games than expected for a season. If any potential issues are found, it prompts the user to delete games from the local database.
