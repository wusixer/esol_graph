"""Module to clean esol data"""

import pandas as pd
from janitor import clean_names, remove_empty, filter_on
from pyprojroot import here

def clean_data(datapath:str) ->pd.DataFrame:
    """
    Clean esol data by replacing column names with 
    space to underscore and remove rows with empty values,
    then filter out rows whose polar_surface_area is <0

    Args:
        datapath (str): relatvie path to esol data

    Returns:
        pd.DataFrame: [description]
    """
    df = (
    pd.read_csv(here()/datapath)
    .pipe(clean_names)
    .pipe(remove_empty)
    .filter_on("polar_surface_area>0")
    )
    
    return df
