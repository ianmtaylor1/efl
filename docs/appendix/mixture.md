---
title: Mixture Distributions
layout: main
---

# Properties of Mixture Distributions

A [mixture distribution](https://en.wikipedia.org/wiki/Mixture_distribution) is
a probability distribution derived from several other distributions which are 
"mixed" together according to selection probabilities. I will derive some
useful properties of mixture distributions (specifically their first and second
moments) for cases that are relevant to the models in this program.

## Definitions

We want to define a random variable $$Y$$ that is a mixture of several random
variables $$X_1, X_2, \dots, X_k$$. For this, we will assume that
$$(X_1,\dots,X_k) = \boldsymbol{X} \sim (\boldsymbol{\mu},\Sigma)$$, i.e. that
the $$X$$'s have mean $$\boldsymbol{\mu} = (\mu_1,\dots,\mu_k)$$ and
(co)variance matrix $$\Sigma = (\sigma_{ij})$$, so they may be correlated.

$$Y = \boldsymbol{S}^\top \boldsymbol{X},$$

