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

This model is an 
[ordinal regression](https://en.wikipedia.org/wiki/Ordinal_regression) model - 
or regression with an ordered outcome. Here the outcome is the game result,
Win, Draw, or Loss, which are naturally ordered from better to worse.

The model is defined by 22 parameters: 
* 20 representing the strength of each of the 20 teams, written
$$s_i$$ for $$i=1,\dots,20$$.
* 1 representing a homefield advantage, written $$h$$.
* 1 determining the relative probability of a draw, written $$\theta$$.

To predict the outcome of a game, you would first calculate the team strength
difference baseline, $$m$$. For example, if Wolves are playing Leicester City
at home,

$$m = h + s_{Wolves} - s_{Leicester}.$$

Then, you would add noise, $$\varepsilon$$ to the baseline to account for the
randomness of the result.

$$y' = m + \varepsilon$$

Here $$\varepsilon$$ is drawn from a 
[Logistic distribution](https://en.wikipedia.org/wiki/Logistic_distribution).
Then $$y'$$ is used to determine the final result, $$y$$:

$$y = \begin{cases}
Loss & y' < -\theta \\
Draw & -\theta \leq y' \leq \theta \\
Win & y' > \theta
\end{cases}$$

## This Model in Code

This model is called efl.model.SymOrdReg in the Python package.

## Priors

The prior on the model's parameters are supposed to (roughly) encapsulate your
prior knowledge of the parameter values - i.e. how good you think each team is
at the start of the season, how much you think homefiled advantage is at the
start of the season, and what you think the probability of a draw is.

In this model, I use a multivariate normal distribution prior on the team 
strength parameter, a normal prior on the homefied advantage parameter, and
logistic distribution priors on the draw boundary parameter.

There are two main ways to set priors: non-informative, and informative.

### Non-informative

The noninformative priors are meant to provide little or no actual prior
information. They are used when you don't know anything about the teams at the
start of the season. In this model, that means a very high variance on the team
strength parameters, a scale of 1 for the draw boundary parameter, and a
standard deviation of $$\approx$$ 1.8 for the homefield advantage parameter.

### Informative

The informative priors on this model are derived from the posterior samples from
this same model in the prior season. The logic is that without any other
information, I would think that teams at the beginning of this season are about
as good as they were at the end of last season. use the posterior means as the
prior means, but inflate the posterior variances by some factor to account for
the fact that we are less certain about each team's ability now at the start
of this season.

This is easy enough for draw boundary and homefield advantage parameters, but
is trickier for the team strength parameters due to promotion and relgation. My
solution to this problem is to assume, for example, that the teams promoted into
the league this year are about as good as the teams relegated out of the league
last year.

Each "promoted in" team's prior strength is the normal distribution whose mean
and variance are the same as a [mixture distributions](../appendix/mixture.html)
of all of the "relegated out" teams, mixed with equal weight.
