# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:38:56 2020

@author: Knaak001
"""

import os
import pandas as pd
import numpy as np
import xarray as xr
import gdal
import pcraster as pcr
from collections import OrderedDict
from matplotlib import pyplot as plt

def unique_values(pcr_map):
    """ Function to retrieve integer values from PCRaster maps.
    
    Input:
        Nominal/Ordinal pcraster map.
    
    Returns:
        array of unique values from input map.
    """
    pcr_map = pcr.scalar(pcr_map)
    ar = pcr.pcr2numpy(pcr_map, np.nan)
    ar = ar[~np.isnan(ar)]
    ar = np.unique(ar).astype(int)
    
    return ar
    
class Pcrmap:
    """Class to get information from and changes datatypes of
       pcraster mapobjects"""
    
    def __init__(self, pcrmap):
        self.path = pcrmap
        self.readmap()
        self.array()
        self.nrows()
        self.ncols()
        self.cellsize()
        self.mapextent()
        pcr.setclone(self.path)
        
    def readmap(self):
        self.map = pcr.readmap(self.path)
    
    def array(self):
        self.array = pcr.pcr2numpy(self.map, np.nan)
    
    def name(self):
        self.name = os.path.basename(self.path)
        return self.name
    
    def minimum(self):
        """Minimum value of the map"""
        return np.nanmin(self.array)
    
    def maximum(self):
        """Maximum value of the map"""
        return np.nanmax(self.array)
          
    def nrows(self):
        self.rows = pcr.clone().nrRows()
    
    def ncols(self):
        self.cols = pcr.clone().nrCols()
    
    def cellsize(self):
        self.cellsize = pcr.clone().cellSize()
    
    def mapextent(self):
        """Bounding box of the map"""
        self.xmin = pcr.clone().west()
        self.xmax = self.xmin + (self.cols*self.cellsize)
        self.ymax = pcr.clone().north()
        self.ymin = self.ymax - (self.rows*self.cellsize)
        return self.xmin, self.xmax, self.ymin, self.ymax
    
    def datatype(self):
        """Return a string of the datatype of the map"""
        mapinfo = gdal.Info(self.path)
        info_list = mapinfo.split('\n')
        valuescale = [i for i in info_list if 'PCRASTER_VALUESCALE' in i][0]
        datatype = valuescale.split('=')[1]
        return datatype
    
    def location_value(self, xco, yco):
        """Return a value of the map at a specified location"""
        xco_map = pcr.xcoordinate(pcr.defined(self.map))
        yco_map = pcr.ycoordinate(pcr.defined(self.map))
        location_map = (xco_map==xco) & (yco_map==yco)
        location_value = unique_values(location_map)[0]
        return location_value
    
    def maphisto(self, bins):
        """Return a histogram values and bin-edges of the map
        with specified bins.
        bins: specify n-number of bins or an array-like of bin-edges
        """
        array = self.array[~np.isnan(self.array)]
        hist, bin_edges = np.histogram(array, bins)
        return hist, bin_edges
    
    def maphisto_plot(self, bins):
        """Plot a histogram of the map with specified bins
        bins: specify n-number of bins or an array-like of bin-edges
        """
        array = self.array[~np.isnan(self.array)]
        hist, bins = np.histogram(array, bins)
        plt.bar((bins[1:] + bins[:-1])*0.5, hist, width=(bins[1]-bins[0])*0.9)
    
    def as_DataArray(self):
        """Create an xarray DataArray of the map"""
        data = xr.open_rasterio(self.path)
        data.values = np.where(data.values==data.values.min(), np.nan,
                               data.values)
        return data
    
class NominalMap(Pcrmap):
    """Class to work on nominal PCRaster maps. Inherits from Pcrmap class"""
    
    def category_count(self):
        """Return a Pandas DataFrame of the number of cells belonging
        to each category of the nominal mapobject"""
        ones = pcr.scalar(pcr.defined(self.map))
        count = pcr.areatotal(ones, self.map)
        data = OrderedDict()
        data['category'] = self.array.flatten()
        data['count_'] = pcr.pcr2numpy(count, np.nan).flatten()
        df = pd.DataFrame(data).\
            dropna().\
            drop_duplicates().\
            sort_values(by='category').\
            reset_index(drop=True)
        return df