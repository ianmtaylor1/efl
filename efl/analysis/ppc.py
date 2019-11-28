"""
ppc.py

Contains code for running posterior predictive checks of models.
"""

import numpy
import matplotlib.pyplot as plt
from . import analysis

class EFL_PPC(object):
    """Class that represents a posterior predictive check of an EFL Model."""
    
    def __init__(self, model, observed, statfun, name=None):
        """Parameters:
            model - a subclass of _EFLModel
            actual - a object of type EFLGames
            statfun - a function which takes games in the form of a dataframe
                and computes some statistic
            name - what to name this statistic. By default it is the name of
                statfun
        """
        # Default name
        if name is None:
            name = statfun.__name__
        self.name = name
        # Convert to data frames (fitted games only)
        # Add team names and dates to predicted games
        pred_df = analysis.add_info(model.to_dataframe("fit"), observed)
        # Compute the statistic for each set of simulated games
        self.ppd = numpy.array([statfun(g) for _,g in pred_df.groupby(['chain','draw'])])
        # Compute the statistic for the observed games
        self.obs = statfun(observed.to_dataframe())
    
    def quantile(self):
        """Return the quantile of the observed games within the posterior
        predictive samples."""
        return numpy.mean(self.ppd <= self.obs)
    
    def plot(self, title=None):
        """Display the posterior predictive samples as a histogram, with a red
        line indicating the observed value."""
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if title is not None:
            fig.suptitle(title)
        ax.hist(self.ppd, density=True)
        ax.axvline(self.obs, color="red")
        ax.ylabel("Frequency")
        ax.xlabel(self.name)
        #plt.hist(self.ppd, density=True)
        #plt.axvline(self.obs, color="red")
        #plt.ylabel("Frequency")
        #plt.xlabel(self.name)
        #if title is not None:
        #    plt.title(title)
        #plt.show()
        return fig

