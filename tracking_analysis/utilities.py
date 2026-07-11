# -*- coding: utf-8 -*-
"""
Created on Fri May 26 14:49:09 2023

@author:    Jonas Hartmann @ Mayor lab @ UCL CDB

@descript:  A collection of convenient helper utilities.
"""

# Imports
import os
import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display


def savebutton(func):
    """Decorator that adds a "save figure" button to a function that generates
    a matplotlib plot, including functions made into interactive widgets using
    ipywidgets.interact. Supports saving as ".png" and ".pdf" (default is pdf).
    
    NOTE: Make sure you don't add `plt.show()` in the figure generation func,
          as that will clear the current figure and save nothing!
          
    Examples
    --------
    
    # Without `interact` widget
    @savebutton
    def make_plot():
        t = np.linspace(0, 10, 500)
        y = np.sin(2.0*t)
        plt.plot(t, y, color='red')
    make_plot()

    # With `interact` widget
    # Note that any defaults in make_plot's kwargs will be overwritten by the
    # defaults of the interact widget.
    @interact(freq=(0.5, 10, 0.5),
              color=['red', 'blue', 'green'])
    @savebutton
    def make_plot(freq=2.0, color='red'):
        t = np.linspace(0, 10, 500)
        y = np.sin(freq*t)
        plt.plot(t, y, color=color)
    """

    def wrapper(**kwargs):
        
        # Prepare textbox (for filename) and button
        textbox = widgets.Text(value='', placeholder='Enter valid file path',
                               description='File path:', disabled=False)
        button  = widgets.Button(description='Save figure!')
        box     = widgets.HBox([textbox, button])
        
        # Callback to save figure when button is clicked
        def on_button_clicked(b):
            if textbox.value:
                figpath = str(textbox.value)
                figdir = os.path.split(figpath)[0]
                if figdir == "":
                    figdir = "."
                    os.path.join(figdir, figpath)
                if not os.path.isdir(figdir):
                    raise ValueError("No directory called "+figdir)
                if figpath.endswith('.pdf'):
                    b.fig.savefig(figpath, bbox_inches='tight', transparent=True)
                elif figpath.endswith('.png'):
                    b.fig.savefig(figpath, dpi=300, bbox_inches='tight', transparent=True)
                    #b.fig.savefig(figpath, dpi=300, bbox_inches='tight', facecolor="w")
                else:
                    raise ValueError("Figure filename must end with '.pdf' or '.png'.")
                print("Saved figure as '%s'" % figpath)
        button.on_click(on_button_clicked)

        # Run wrapped function to generate figure
        func(**kwargs)
        
        # Update figure in button
        button.fig = plt.gcf()
        
        # Display textbox and button
        display(box)
        
    # Done!
    return wrapper  


def get_shifted_data(df, col_1, col_2, shift):
    """Get track-matched but frame-shifted data from two specified columns of a 
    standard tracking data df. The extracted data from the 2nd column is offset 
    from that in the 1st column by `shift` number of frames.
    
    Parameters
    ----------
    df : pandas df; index:'track', columns:(f, t, y, x, ...)
        The df containing the tracking data. Must be structured in the standard
        way expected in tracking_analysis pipelines.
    col_1 : str
        First column from which values will be extracted, e.g. 'v_mag'.
    col_2 : str
        Second column from which values will be extracted *after frame-shifting*
        by `shift` number of frames, e.g. 'v_mag_itp'.
    shift : int
        Number of frames by which data in `col_2` is shifted. Can be negative.
    
    Returns
    -------
    col_1_data : numpy array, 1d
        Data from `col_1` of `df`.
    col_2_data : numpy array, 1d, same shape as col_1_data
        *Frame-shifted* data from `col_2` of `df`.
    """
    
    # Get relevant data
    df_c1 = df[['f', col_1]].copy()
    df_c2 = df[['f', col_2]].copy()
    
    # Introduce the shift for column 2
    df_c2.loc[:, 'f'] -= shift
    
    # Remerge the two columns
    merged = pd.merge(df_c1, df_c2, on=["track", "f"])
    
    # Return values
    return merged[col_1].values, merged[col_2].values

