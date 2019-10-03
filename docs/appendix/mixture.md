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
$$\boldsymbol{X} = (X_1,\dots,X_k)^\top \sim (\boldsymbol{\mu},\Sigma)$$, i.e.
that the $$X_i$$'s have mean $$\boldsymbol{\mu} = (\mu_1,\dots,\mu_k)^\top$$ 
and (co)variance matrix $$\Sigma_{k\times k} = (\sigma_{ij})_{i,j=1,\dots,k}$$,
so they may be correlated.

With that, we define

$$Y = \boldsymbol{S}^\top \boldsymbol{X},$$

where $$\boldsymbol{S} \sim multinomial(1; \boldsymbol{p})$$ is the random
selection vector with a 1 in exactly one element, 0's everywhere else. The
vector $$\boldsymbol{p} = (p_1,\dots,p_k)^\top$$ are the selection
probabilities: $$p_i$$ is the probability that $$S_i = 1$$ and $$Y = X_i$$.