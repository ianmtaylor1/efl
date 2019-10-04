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

where
$$\boldsymbol{S} = (S_1,\dots,S_k)^\top \sim multinomial(1; \boldsymbol{p})$$
is the random selection vector with a 1 in exactly one element and 0's
everywhere else. The vector $$\boldsymbol{p} = (p_1,\dots,p_k)^\top$$ are the
selection probabilities: $$p_i$$ is the probability that $$S_i = 1$$ and 
$$Y = X_i$$.

It is important to note that $$\boldsymbol{X}$$ and $$\boldsymbol{S}$$ are
independent ($$\boldsymbol{X} \perp \boldsymbol{S}$$), i.e. which variable is
selected is independent of the variables' values.

## Mean of a Mixture Distribution

$$
\begin{align*}
E[Y] &= E\left[\sum_{i=1}^k S_i X_i\right] \\
&= \sum_{i=1}^k E[S_i X_i] \\
&= \sum_{i=1}^k E[S_i]E[X_i] & \text{due to independence} \\
&= \sum_{i=1}^k p_i \mu_i \\
&= \boldsymbol{p}^\top \boldsymbol{\mu}
\end{align*}
$$

## Variance of a Mixture Distribution

First, write

$$
\begin{align*}
Var(Y) &= E[Y^2] - E[Y]^2 \\
&= E[Y^2] - (\boldsymbol{p}^\top \boldsymbol{\mu})^2.
\end{align*}
$$

Now consider the $$E[Y^2]$$ term:

$$
\begin{align*}
E[Y]^2 &= E\left[\left(\sum_{i=1}^k S_i X_i\right)^2\right] \\
&= E\left[\sum_{i=1}^k S_i^2 X_i^2\right],
\end{align*}
$$

since all of the "cross terms" will have a
product $$S_i S_j$$, with $$i \neq j$$. These necessarily equal zero, since at
least one of $$S_i$$ or $$S_j$$ equals zero. (Only one component of
$$\boldsymbol{S}$$ equals 1.) Then,

$$
\begin{align*}
E[Y^2] &= E\left[\sum_{i=1}^k S_i^2 X_i^2\right] \\
&= \sum_{i=1}^k E[S_i^2 X_i^2] \\
&= \sum_{i=1}^k E[S_i^2]E[X_i^2] & \text{due to independence} \\
&= \sum_{i=1}^k p_i(Var(X_i) + E[X_i]^2),
\end{align*}
$$

since $$S_i^2 = S_i$$ and $$Var(X_i) = E[X_i^2] - E[X_i]^2$$. Finally,

$$E[Y^2] = \sum_{i=1}^k p_i(\sigma_{ii} + \mu_i^2),$$

so

$$Var(Y) = \sum_{i=1}^k p_i(\sigma_{ii} + \mu_i^2) - 
(\boldsymbol{p}^\top \boldsymbol{\mu})^2$$

## Covariance of a Mixture Distribution

### ...with an arbitrary random variable

Define a random variable $$Z$$ such that $$Z \perp \boldsymbol{S}$$ but $$Z$$ may be correlated with $$\boldsymbol{X}$$.

$$
\begin{align*}
Cov(Y,Z) &= Cov\left(\sum_{i=1}^k S_i X_i, Z\right) \\
&= \sum_{i=1}^k Cov(S_i X_i, Z) \\
&= \sum_{i=1}^k \left(E[S_i X_i Z] - E[S_i X_i]E[Z]\right) \\
&= \sum_{i=1}^k \left(E[S_i X_i Z] - p_i \mu_i E[Z]\right) \\
&= \sum_{i=1}^k \left(E[S_i]E[X_i Z] - p_i \mu_i E[Z]\right) 
& \text{due to independence}\\
&= \sum_{i=1}^k \left(p_i (Cov(X_i, Z) + E[X_i]E[Z]) - p_i \mu_i E[Z]\right) \\
&= \sum_{i=1}^k p_i Cov(X_i, Z) \\
&= \boldsymbol{p}^\top Cov(\boldsymbol{X}, Z)
\end{align*}
$$

### ...with another mixture of the same variables $$X_i$$

Consder two mixture random variables 
$$Y_1 = \boldsymbol{S}_1^\top \boldsymbol{X}$$ and
$$Y_2 = \boldsymbol{S}_2^\top \boldsymbol{X}$$. As before, 
$$\boldsymbol{S}_1 \perp \boldsymbol{X}$$ and
$$\boldsymbol{S}_2 \perp \boldsymbol{X}$$. Also assume
$$\boldsymbol{S}_1 \perp \boldsymbol{S}_2$$.

The situation described is that $$Y_1$$ and $$Y_2$$ are independent _mixtures_
of the same random variables $$X_1,\dots,X_k$$. The selections are independent,
but there is a non-zero probability that $$Y_1 = Y_2$$, since if 
$$\boldsymbol{S}_1 = \boldsymbol{S}_2$$ both will select the same $$X_i$$.

$$
\begin{align*}
Cov(Y_1,Y_2) &= Cov\left(\sum_{i=1}^k S_{1i}X_i, \sum_{j=1}^k S_{2j}X_j\right)\\
&= \sum_{i=1}^k \sum{j=1}^k Cov(S_{1i}X_i, S_{2j}X_j) \\
&= \sum_{i=1}^k \sum{j=1}^k \left(E[S_{1i}X_i S_{2j}X_j] 
                                  - E[S_{1i}X_i]E[S_{2j}X_j]\right) \\
&= \sum_{i=1}^k \sum{j=1}^k \left(p_i p_j E[X_i X_j] - p_i p_j\mu_i\mu_j\right)
& \text{due to independence} \\
&= \sum_{i=1}^k \sum{j=1}^k \left(p_i p_j(Cov(X_i,X_j) + E[X_i]E[X_j]) 
                                  - p_i p_j\mu_i\mu_j\right) \\
&= \sum_{i=1}^k \sum{j=1}^k p_i p_j\sigma_{ij} \\
&= \boldsymbol{p}^\top \Sigma \boldsymbol{p}
\end{align*}
$$
