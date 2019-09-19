---
title: Symmetric Ordinal Regression
layout: main
---

# Symmetric Ordinal Regression

This is the first and the simplest model that I wrote. The model centers around
a "strength" for each team. The difference between two teams' strengths
determines the probability of a win, draw, or loss. For example, if Wolves'
strength is much higher than Leicester City's strength, Wolves will have a high
proability of a win, and a lower probability of a draw or a loss.

Home teams are given a bonus to their strength, known as homefield advantage.
So Wolves will have a higher probability of winning at home against Leiceter
than they would away.

Two teams with approximately equal strengths will have approximately equal
probability of winning, and a larger probability of a draw than two teams with
unequal strengths.

## Technical Definition and Notation

This model is an ordinal regression model - or regression with an ordered
outcome. Here the outcome is the game result, Win, Draw, or Loss, which
are naturally ordered from better to worse.

The model is defined by 22 parameters: 
* 20 representing the strength of each of the 20 teams, written
$$s_i$$ for $$i=1,\dots,20$$.
* 1 representing a homefield advantage, written $$h$$.
* 1 determining the relative probability of a draw, written $$\theta$$.

## This Model in Code

This model is called efl.model.SymOrdReg in the Python package.
