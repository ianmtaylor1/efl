---
title: EFL Model Details
layout: main
---

# EFL Model Details

This page contains descriptions of the different models implemented in this package. The models are broadly divided into two categories: Goal Models and Result Models. Goal Models are those which attempt to predict the score of a game, while Result Models only attempt to predict the result (win, draw, loss). Generally, Goal Models are based on [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression) and Result Models are based on [ordinal regression](https://en.wikipedia.org/wiki/Ordered_logit), though the details vary from model to model.

## Result Models

* [Symmetric Ordinal Regression](symordreg.html)
* Symmetric Ordinal Regression HTI (home/team interaction)

## Goal Models

* [Numberphile Model](numberphile.html)
* Simplified Model
