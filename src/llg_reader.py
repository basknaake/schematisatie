# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:16:44 2020

@author: Knaak001
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.geometry import box
from tqdm import tqdm
tqdm.pandas()

def thickness_sand_layers(df):
    """
    Determine the thickness of sand layers and label all layers with a 
    unique ID in a dataframe of a core

    Parameters
    ----------
    df: DataFrame
        Pandas DataFrame of the core 

    Returns
    -------
    core: DataFrame
        Pandas DataFrame of the core
    """
    core = df.copy()
    ## create column of 0/1 for nosand/sand
    core['sand'] = None
    core.loc[core['textuur'].str[0]=='z', 'sand'] = 1
    core['sand'] = core['sand'].fillna(0)

    ## give ID to consecutive sand layers
    core['layer_ID'] = (core['sand'] != core['sand'].shift(1)).cumsum()
    
    ## calculate difference between the layers to determine the amount of layers of 10cm
    diff_layers = np.diff(core.diepte)
    factor = np.insert(diff_layers, 0, core.diepte.iloc[0])/10
    core['factor'] = factor.astype(int)
    factor_len = core['factor'].sum() # number of 10cm layers in core
    
    ## determine top of each layer
    core['top'] = core['diepte']-(factor*10)
    core['top'] = core['top'].astype(int)
    
    core = core.loc[np.repeat(core.index.values, core.factor)]
    core['new_depth'] = [i for i in range(core['top'].iloc[0]+10,
                                          core['top'].iloc[0]+\
                                              (factor_len.sum()*10)+1, 10)]
    
    ## determine the thickness of all layers
    thickness = core.groupby('layer_ID')['new_depth'].\
        apply(lambda x: x.max()-x.min()+10).\
        reset_index(name='thickness')
    
    ## add thickness to core as new column
    core = pd.merge(core, thickness, on='layer_ID', how='left')
    
    ## create good output
    outcols = ['top', 'thickness', 'text', 'm50', 'sand', 'layer_ID']
    core = core.drop_duplicates(subset=['layer_ID', 'm50'])
    core['text'] = np.where(core.sand==1, 'Sand', 'Other')
    core = core[outcols]
    core = core.sort_values(by='top')
    
    return core

class ReadLLG:
    """
    Class to read and interpet LLG data, consists of "Hoofd" and "Data"
    tables. "Hoofd" table is opened as a GeoDataFrame and "Data" table
    is opened as a DataFrame.
    """
    def __init__(self, path):
        self.path = path
        self.readHoofd()
        self.readData()
        
    def readHoofd(self):
        hoofd = os.path.join(self.path, 'llg_hoofd.txt')
        dtypes = {'GWT':str} # GWT column has variable datatypes 
        self.hoofd = pd.read_csv(hoofd, dtype=dtypes)
        self.hoofd.columns = map(str.lower, self.hoofd.columns)
        self.hoofd['geometry'] = [Point(x, y) for x, y in
                                  zip(self.hoofd['xco'], self.hoofd['yco'])]
        
        self.hoofd = gpd.GeoDataFrame(self.hoofd, geometry='geometry')
        
    def readData(self):
        data = os.path.join(self.path, 'llg_data.txt')
        dtypes = {'RE':str, 'OG':str, 'CA':str, 'M':str, 'STRAT':str}
        self.data = pd.read_csv(data, dtype=dtypes)
        self.data.columns = map(str.lower, self.data.columns)
    
    def select_core_hoofd(self, core_ID):
        """Select data for one or more specific cores
        core_ID: array_like
            Array or list of IDs of the cores to select
        """
        core = self.hoofd.loc[self.hoofd['boorp'].isin(core_ID)]
        return core
    
    def select_core_data(self, core_ID):
        """Select data for one or more specific cores
        core_ID: array_like
            Array or list of IDs of the cores to select
        """
        core = self.data.loc[self.data['boorp'].isin(core_ID)]
        return core
    
    def select_with_polygon(self, polygon):
        """ Mask DataFrame with a polygon
        polygon: Shapely Polygon or MultiPolygon object
        """
        to_mask = gpd.GeoDataFrame(geometry=[polygon])
        masked = gpd.sjoin(self.hoofd, to_mask, how='inner', op='within')
        
        data = self.data.loc[self.data['boorp'].isin(masked.boorp.unique())]
        return data
    
    def sand_layers(self):
        """Return LLG data table grouped as sand layers with an ID and
        corresponding thickness
        """
        layered = self.data.groupby('boorp').\
            progress_apply(thickness_sand_layers)
        return layered
    


