#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats.py

Contains functions which compute statistics on sets of games (simulated or
real). Useful for the statfun argument in the ppc function.
"""

def homewins(g):
    """Calculates the homefield advantage: what proportion of games in the
    DataFrame g were won by the home team."""
    return sum(g['result'] == 'H')/g['result'].count()

def draws(g):
    """Calculates the likelihood of a draw: what proportion of games in the
    DataFrame g resulted in a draw."""
    return sum(g['result'] == 'D')/g['result'].count()

