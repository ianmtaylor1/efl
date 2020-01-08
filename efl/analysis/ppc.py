#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppc.py

File containing a class for conducting posterior predictive checks of
a model. It is based on an EFLPredictor, but highlights and compares the stat
values on the observed data with what is generated from the model predictions.
"""

import numpy
import pandas

from . import predictor

class EFLPPC(predictor.EFLPredictor):
    """EFLPPC - class built to conduct posterior predictive checks on a
    model."""
    
    # Modify init to add places for observed data and statistics
    
    def __init__(self, model, mode="past", **kwargs):
        # Initialize the base object
        super().__init__(model, mode, **kwargs)
        # Store the observed dataframe of games
        if mode == "past":
            self._observed_data = model.gamedata.loc[model.fitgameids,:]
        elif mode == "future":
            self._observed_data = model.gamedata.loc[model.predictgameids,:]
        elif mode in ["full", "mix"]:
            self._observed_data = model.gamedata
        # A place to store statistics calculated on observed data
        self._observed_values = {}
        
    # Modify _compute_stat to calculate values for observed data
    
    def _compute_stat(self, stat, *args, **kwargs):
        # Do everything with the base object
        super()._compute_stat(stat, *args, **kwargs)
        # Compute the observed values of this stat
        if self._stat_precompute.get(stat, None) is None:
            self._observed_values[stat] = stat(self._observed_data)
        else:
            params = (self._observed_values[p] for p in self._stat_precompute[stat])
            self._observed_values[stat] = stat(*params)
    
    # Modify to_dataframe to add a row for the observed stat values
    
    def to_dataframe(self, names=None, *args, **kwargs):
        # Get the dataframe without the observed values
        df = super().to_dataframe(names, *args, **kwargs)
        # Convert stat parameter to list format, defaulting to all
        if names is None:
            names = self.names
        elif type(names) == str:
            names = [names]
        # Make a series with the observed values
        obsdata = {('chain',''):['observed'], ('draw',''):['']}
        for n in names:
            if n in self._name2stat:
                obsdata[(n,'')] = [self._observed_values[self._name2stat[n]]]
            elif n in self._groups:
                for sn,ss in self._groups[n].items():
                    obsdata[(n,sn)] = [self._stat_values[ss]]
        obsrow = pandas.DataFrame(obsdata).set_index(['chain','draw'])
        # Add it to the data frame and return it
        return pandas.concat([obsrow, df], sort=True)
    
    # Modify plotting internals to highlight the observed values
    
    def _plot_numeric(self, stat, name, ax, *args, **kwargs):
        # Draw the base histogram plot on the axis
        hist = super()._plot_numeric(stat, name, ax, *args, **kwargs)
        # Draw a vertical line where the observed value is
        leq = 100.0 * (
                numpy.array(self._stat_values[stat]) 
                <= self._observed_values[stat]
                ).mean()
        lab = "Observed %ile: {:.0f}".format(leq)
        ax.axvline(self._observed_values[stat], color="red", label=lab)
        ax.legend()
        # Pass through the original return value
        return hist
    
    def _plot_categorical(self, stat, name, ax, *args, **kwargs):
        # Draw the base bar plot on the axis
        bars = super()._plot_categorical(stat, name, ax, *args, **kwargs)
        # Figure out which bar needs to be highlighted
        hltidx = None
        labels = [l.get_text() for l in ax.get_xticklabels()]
        if str(self._observed_values[stat]) in labels:
            hltidx = labels.index(str(self._observed_values[stat]))
        # Hightlight that bar
        if hltidx is not None:
            bars[hltidx].set_color("red")
            bars[hltidx].set_label("Observed Value")
            ax.legend()
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xpos = xlim[0] + 0.92*(xlim[1] - xlim[0])
            ypos = ylim[0] + 0.92*(ylim[1] - ylim[0])
            t = ax.text(xpos, ypos,
                        "Observed: {}".format(self._observed_values[stat]),
                        fontsize=12, ha="right", va="top")
            t.set_bbox(dict(boxstyle='square', edgecolor="red", 
                            facecolor="white", pad=0.3))
        # Return the original value
        return bars
    
    