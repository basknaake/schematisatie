# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:10:56 2020

@author: Knaak001
"""

import numpy as np
import pandas as pd
import xarray as xr
import pcraster as pcr
from tqdm import tqdm

def calc_percentage_array_items(array, remove):#, stratcodes):
    """
    Create a Pandas Series with calculated relative percentages of each item
    by dividing all values by the sum of values in an nD-array

    Parameters
    ----------
    array : array
        array to calcultate the percentages of the items
    remove: list
        list of items to remove from array that must not be
        included in calculation

    Returns
    -------
    s_perc : Series
        Pandas series with percentages of the items of the input array
    """
    s = pd.Series(array.flatten())
    s = s[~s.isin(remove)] # remove specified items from series
    s_perc = s.value_counts(normalize=True)
    
    return s_perc

def add_index_info(series, ID, stratcodes):
    """
    Add location and geologic information to the index of a Pandas Series

    Parameters
    ----------
    series : Series
        Pandas Series to add the information to
    ID : int
        ID value (e.g. x-coordinate) to add to the index
    stratcodes: DataFrame
        Pandas dataframe with information of the stratigraphic units in
        GeoTop

    Returns
    -------
    series : Series
        Pandas Series with the added information

    """
    groups = stratcodes[['nr', 'type']].drop_duplicates()
    
    ## replace index with strat groups and sum values if index is duplicate
    series.index = [groups.loc[groups['nr']==b, 'type'].iloc[0]
                    for b in series.index]
    series = series.sum(level=0)
    
    ## add ID to index
    series.index = pd.MultiIndex.from_tuples([(ID, b) for b in series.index])
    
    return series

def select_data_with_pcr(geo_data, pcrmap):
    """
    Select the GeoTop cells that are within the area where the pcrmap has
    data
    
    Parameters
    ----------
    geo_data : DataArray
        Attribute of GeoTop voxelmodel as xarray DataArray
    attribute: string
        GeoTop attribute to select the data for
    pcrmap: PCRaster
        PCRaster map object of the area to select. Required cellsize of the 
        map is 100x100 meters and cellcentre is matched to GeoTop cellcentre
    
    Returns
    -------
    river_data: DataArray
        Xarray DataArray containing xyz data and corresponding attribute
        for all voxels for the river area of GeoTop 
    """
    ## create boolean array, true for cells that belong to rivers
    arr = pcr.pcr2numpy(pcr.defined(pcrmap), np.nan) == 1
    
    ## match dimensions of array to geotop
    arr = np.flip(arr[:,:1475], axis=0)
    
    ## create 3D boolean river array
    temp_ar = arr.copy()
    for i in tqdm(range(geo_data.shape[2]-1),desc='create selection array'):
        arr = np.dstack((arr, temp_ar))
    
    ## select data from lithostrat within rivers
    select = geo_data.where(arr)
    
    return select

def proportion_3D_over_xco(geo_data, stratcodes, drop=[], top=None,
                           bottom=None, intervals=None):
    """
    Calculate 3-dimensional proportional contribution of stratigraphic units
    based on geotop data in a specified depth window below the surface

    Parameters
    ----------
    geo_data : DataArray
        Attribute of GeoTop voxelmodel as xarray DataArray
    stratcodes: DataFrame
        Pandas dataframe with information of the stratigraphic units in
        GeoTop
    drop: list
        list of items to must not be included in calculation of proportion
    top: int, float, optional
        Define top of the depth interval incorporated in the calculation.
        The default is None.
    bottom : int, float, optional
        Define bottom of the depth interval incorporated in the calculation.
        The default is None.
    intervals: DataFrame
        Pandas DataFrame with upper and lower 

    Returns
    -------
    table : DataFrame
        Pandas dataframe with proportions of stratigraphic groups per
        x-coordinate
    """
    ## array of coordinates 
    xcoords = geo_data.x.values
    
    ## select depth interval
    geo_data = geo_data.sel(z=slice(bottom, top))
    
    ## initiate first series to append data to
    total = pd.Series()
    
    ## loop over x-coordinates and calculate proportion of units
    for ID in tqdm(xcoords, desc=f'Calculate proportion {geo_data.name}'):
        arr = geo_data.sel(x=ID)
        
        if intervals is not None: 
            depth = intervals.loc[intervals['xco']==ID,
                                  ['lower', 'upper']].values[0]
            arr = arr.sel(z=slice(depth[0], depth[1]))
        
        perc = calc_percentage_array_items(arr.values, drop)
        try:
            perc = add_index_info(perc, ID, stratcodes)
        except:
            continue # continue if series is empty
        total = total.append(perc)
    
    total.index = pd.MultiIndex.from_tuples(total.index)
    table = total.unstack().fillna(0)
  
    return table

def round_to(n, base=5):
    """
    Round value (n) to closest base
    """
    return base * round(n/base)

def round_to_lower(n, base):
    """
    Round value (n) to lower base
    """
    return n - (n % base)

def match_bbox_to_geotop(branch_gdf):
    """
    Match the bounding box of a branch polygon to GeoTop cells

    Parameters
    ----------
    branch_gdf: GeoDataFrame
        Geopandas dataframe containing the polygon of the branch

    Returns
    -------
    left, bottom, right, top: int
        Integer values of the bounding box
    """
    pol = branch_gdf.geometry.iloc[0]
    left, bottom, right, top = pol.bounds
    
    ## match bounding box
    left = left = round_to_lower(left, 100) + 50
    bottom = round_to_lower(bottom, 100) + 50
    right = round_to_lower(right, 100) + 50
    top = round_to_lower(top, 100) + 50
    
    return left, bottom, right, top

def depth_interval_over_xco(df_depths, attr, upper, lower):
    """
    Create depth intervals per x-coordinate based on mean depth of 
    bathymetry to determine the upper and lower boundaries to select
    GeoTop data for that x-coordinate

    Parameters
    ----------
    df_depths: DataFrame 
        Pandas DataFrame with x-coordinates and mean depth 
    attr: string
        min_/mean_, specify if minimum or mean depth is considered
    upper: int
        Specify the amount of meters that will be selected above
        mean depth
    lower: int
        Specify the amount of meters that will be selected below
        mean depth

    Returns
    -------
    df_depths : DataFrame
        Pandas DataFrame with lower and upper boundaries per x-coordinates
    """
    df_depths['rounded'] = round_to(df_depths[attr], 0.5) # round 0.5 meters
    
    df_depths['upper'] = df_depths['rounded'] + upper
    df_depths['lower'] = df_depths['rounded'] - lower
    
    return df_depths

def cmap_units_geotop(units, codes):
    """
    Helper function to create matplotlib colormap for plots with GeoTop
    units
    
    Parameters
    ----------
    units: Array like
        Units to select rgb hextriplets to create colormap for
    codes: DataFrame
        Pandas dataframe with information of the stratigraphic units in
        GeoTop
    
    Returns
    -------
    cmap: list
        List with rgb hextriplets in correct order for matplotlib plot
    """
    codes = codes[['type', 'rgb_hex']].drop_duplicates()
    rgb_dict = {a:b for a, b in zip(codes.type, codes.rgb_hex)}    
    cmap = [rgb_dict[a] for a in units]
    
    return cmap

def map_to_dataarray(mappath):
    """
    Create a xarray DataArray from an input map

    Parameters
    ----------
    mappath: string
        Path to the mapfile to create a DataArray from

    Returns
    -------
    data: DataArray
        Xarray DataArray of the input map
    """
    data = xr.open_rasterio(mappath)
    
    ## replace nodata values with numpy nan
    data.values = np.where(data.values==data.values.min(), np.nan,
                           data.values)
    
    return data

def highest_index(arr):
    """
    Helper function to return the highest index of non-NaN values in 1D array
    """
    try:
        return np.max(np.where(~np.isnan(arr)))
    except ValueError:
        return np.nan

def top_depth_data(geo_data):
    """
    Create a 2D DataArray with the xy extent of the input DataArray
    with labels of the highest cells with data

    Parameters
    ----------
    geo_data: DataArray
        3-dimensional xarray DataArray

    Returns
    -------
    top_depth: DataArray
        2-dimensional xarray DataArray with the labels of the highest
        cells that contain data
    """
    ## create DataArray of the highest index where input data is not NaN
    indexes = np.apply_along_axis(highest_index, 2, geo_data)
    indexes = xr.DataArray(indexes, coords=[geo_data.y, geo_data.x],
                           dims=['y', 'x'])
    
    ## create DataArray of the depth labels corresponding to the indexes
    top_depth = geo_data[:, :, indexes].z
    
    return top_depth

def geotop_mv_map(geo_data, inmap, depth=0):
    """
    Read GeoTop data at, or a depth below maaiveld
    
    Parameters
    ----------
    geo_data : DataArray
        Attribute of GeoTop voxelmodel as xarray DataArray
    inmap: string
        Path of the map to mask the dataframe with
    depth: float
        depth beneath maaiveld, defaults to 0
        
    Returns
    -------
    mv_arr: DataArray
        2D DataArray of the unit at the requested depth
    """
    xr_map = map_to_dataarray(inmap)
    xr_map = xr_map[0]
    
    ## select the area of the input map
    area_sel = geo_data.sel(x=xr_map.x, y=xr_map.y, method='nearest')
    
    top_data = top_depth_data(area_sel)
    
    ## create data array of depths to select
    to_select = (round_to_lower(xr_map, 0.5) + 0.25) - depth
    to_select.values = np.where((np.isnan(to_select.values))|\
                                (to_select.values>top_data.values),
                                top_data, to_select.values)    
    
    np.warnings.filterwarnings('ignore')
    mv_arr = area_sel.loc[:, :, to_select]
    mv_arr.values = np.where(np.isnan(xr_map.values), np.nan, mv_arr.values)    
    
    return mv_arr

def xarray_to_pcraster(data):
    """
    Create a scalar PCRaster map object from an xarray DataArray

    Parameters
    ----------
    data: DataArray
        xarray DataArray to create PCRaster map from

    Returns
    -------
    outmap: PCraster
        PCRaster map object from the DataArray
    """
    outmap = pcr.numpy2pcr(pcr.Scalar, data.values, np.nan)
    
    return outmap