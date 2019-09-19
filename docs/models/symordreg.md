---
title: Symmetric Ordinal Regression
layout: main
---

# Symmetric Ordinal Regression

This is the first model that I wrote and by far the simplest. The main idea is that each team has a "strength", and the difference between two teams' strengths determined the probability of a win, draw, or loss.

More specifically, it involves 22 parameters: 20 representing the strength of each of the 20 teams ($$s_i$$ for $$i=1,\dots,20$$), one representing a homefield advantage ($$h$$), and one determining the relative probability of a draw ($$\theta$$).
