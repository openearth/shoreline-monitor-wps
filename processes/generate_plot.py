#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2025 Deltares
#     Irham Adrie Hakiki
#
#   This library is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This library is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this library.  If not, see <http://www.gnu.org/licenses/>.
#   --------------------------------------------------------------------
#
# This tool is part of <a href="http://www.OpenEarth.eu">OpenEarthTools</a>.
# OpenEarthTools is an online collaboration to share and manage data and
# programming tools in an open source, version controlled environment.
# Sign up to recieve regular updates of this function, and to contribute
# your own tools.

import matplotlib

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import os
from shapely.geometry import box, Point, Polygon, LineString
from sklearn.linear_model import LinearRegression
import geopandas as gpd
import holoviews as hv
import hvplot.pandas
import colorcet as cc
import numpy as np
from pywps.app.exceptions import ProcessError
import time
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
import panel as pn
import logging

hv.extension("bokeh")
matplotlib.use('Agg')   # Use a non-GUI backend for PNG output
logger = logging.getLogger("PYWPS")


def PostProcShoreline(transect_name,df,start=None,end=None):
    """Post-processing shoreline series data into 3 plots:
       1. Time series of selected transect compared to the neighbor
       2. Time stack of shoreline position
       3. Time stact of shoreline change

    Args:
        transect_name: name of the selected transect
        df: dataframe of shoreline monitoring series

        optional (need to modify the front end to receive date input)
        start: start date
        end: end date

    Returns:
        new_rate_df: name of corresponding transect
        plot: list of plot object
    """ 
    
    # filtering the dataframe with primary observation
    df_prim = df[df['obs_is_primary']]

    # validating date input
    if not start and not end:
        date_filter = df_prim
    else:
        date_filter = df_prim[
            (df_prim["datetime"] >= start if start is not None else True) &
            (df_prim["datetime"] <= end if end is not None else True)
        ]

    # calculate a new rate if start
    if start or end:
        logger.info(f"=== Calculating new rate ===")
        new_rate_df = CalculateNewRate(date_filter)
    else:
        new_rate_df = pd.DataFrame(df['transect_id'].unique(),columns=['transect_id'])

    # create a table of time and shoreline position
    pivot = date_filter.pivot_table(
        index='transect_id', 
        columns='datetime', 
        values='shoreline_position', 
        aggfunc='mean' 
    )

    pivot = pivot.sort_index(ascending=False)

    # find the location of the selected transect
    selected_transect = pivot.index.get_loc(transect_name)

    # replace transect information to the relative distance from the first data
    dist = np.arange(0,len(pivot)*100,100)
    pivot.index = dist

    # call to each plotting function
    logger.info(f"=== Creating time series plot ===")
    plot1 = PlotSelectedTimeSeries(pivot,selected_transect)
    logger.info(f"=== Creating time stack plot of shoreline position ===")
    plot2 = PlotTimeStackPosition(pivot)
    logger.info(f"=== Creating time stack plot of annual shoreline change ===")
    plot3 = PlotTimeStackDifference(pivot)

    plot = [plot1,plot2,plot3]

    return new_rate_df, plot

def CalculateNewRate(df):
    """Function for calculating new ambient rate based on filtered dataframe

    Args:
        df: dataframe of shoreline monitoring series that have been date filtered

    Returns:
        new_rate_df: dataframe with the transect name (and new ambient rate)
    """ 

    # perform linear regression fit for each transect
    new_df = []
    for id_val, group in df.groupby('transect_id'):
        try:
            X = (
                group["datetime"].dt.year.values
                - group["datetime"].dt.year.values[0]
            )
            y = group.shoreline_position.values
            model = LinearRegression().fit(X.reshape(-1, 1), y)
            # x_values = group["datetime"].dt.year.values
            # y_values = model.intercept_ + model.coef_[0] * X
            slope = round(model.coef_[0], 2)

        except Exception:  
            slope = np.nan

        new_df.append({
            "transect_id": id_val,
            "sds_change_rate": slope,
        })


    return pd.DataFrame(new_df)

def PlotTimeStackPosition(pivot):
    """Plotting table of spatio-temporal shoreline position:

    Args:      
        pivot: table of time and shoreline position

    Returns:
        plot: plot object
    """ 

    fig, ax = plt.subplots(figsize=(10, 8))

    # Set the plot range considering the quantiles
    lower_quantile = pivot.min().quantile(0.025)
    upper_quantile = pivot.max().quantile(0.975)
    min_range = min(lower_quantile, -15)
    max_range = max(upper_quantile, 15)

    # Ensure that the color limits are symmetric around zero
    max_abs_range = max(abs(min_range), abs(max_range))
    clim = (-max_abs_range, max_abs_range)

    cmap = cc.m_diverging_gwr_55_95_c38.copy()
    cmap.set_bad(color="gray")


    c = ax.pcolormesh(#pivot,
        pivot.index,             # datetime axis
        pivot.columns,           # transect axis
        pivot.values.T,            # grid of values
        cmap=cmap,           # color map
        clim=clim,
        shading='auto',
        
    )

    # plot option
    ax.set_ylabel("Year")
    ax.set_xlabel("Distance along Coastline (m)")
    ax.set_title("Spatio-temporal of Shoreline Position from 2023 Reference Line")
    fig.colorbar(c, ax=ax, label="Shoreline Position (m)")
    ax.yaxis.set_major_locator(mdates.YearLocator())
    step = np.round(pivot.index[-1]/1000)*1000/10
    ax.set_xticks(np.arange(min(pivot.index), max(pivot.index), step))
    plt.tight_layout()
    plt.close()

    logger.info(f"=== Time stack plot of shoreline position succesfully created ===")
    return fig

def PlotTimeStackDifference(pivot):
    """Plotting table of spatio-temporal annual shoreline change:

    Args:      
        pivot: table of time and shoreline position

    Returns:
        plot: plot object
    """ 
    
    # Calculate annual change
    pivot_change = pivot.diff(axis=1)
    
    # Set the plot range considering the quantiles
    lower_quantile = pivot_change.min().quantile(0.025)
    upper_quantile = pivot_change.max().quantile(0.975)
    min_range = min(lower_quantile, -15)
    max_range = max(upper_quantile, 15)

    # Ensure that the color limits are symmetric around zero
    max_abs_range = max(abs(min_range), abs(max_range))
    clim = (-max_abs_range, max_abs_range)


    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = cc.m_diverging_gwr_55_95_c38.copy()
    cmap.set_bad(color="gray")

    cdif = ax.pcolormesh(
        pivot_change.index,             # datetime axis
        pivot_change.columns,           # transect axis
        pivot_change.values.T,            # grid of values
        cmap=cmap,           # color map
        clim=clim,
    )

    # plot option
    ax.set_ylabel("Year")
    ax.set_xlabel("Distance along Coastline (m)")
    ax.set_title("Spatio-temporal of Yearly Shoreline Change")
    fig.colorbar(cdif, ax=ax, label="Shoreline Change (m)")
    ax.yaxis.set_major_locator(mdates.YearLocator())
    step = np.round(pivot.index[-1]/1000)*1000/10
    ax.set_xticks(np.arange(min(pivot.index), max(pivot.index), step))
    plt.tight_layout()
    plt.close()

    logger.info(f"=== Time stack plot of annual shoreline change succesfully created ===")
    return fig


def PlotSelectedTimeSeries(pivot,selected_transect):
    """Plotting table of spatio-temporal annual shoreline change:

    Args:      
        pivot: table of time and shoreline position

    Returns:
        plot: plot object
    """ 
    
    fig, ax = plt.subplots(figsize=(10, 5))

    # plot every individual transect and highlight the selected transect
    for i in range(len(pivot)):
        if i == 0:
            ax.plot(pivot.columns, pivot.iloc[i],c='lightgrey',label='Neighbouring Transect')
        
        if i == selected_transect:
            continue

        ax.plot(pivot.columns, pivot.iloc[i],c='lightgrey')

    ax.plot(pivot.columns, pivot.iloc[selected_transect], color='red',label='Selected Transect',marker='o')

    # plot option
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=90)
    plt.xlim([pivot.columns.min(),pivot.columns.max()])
    plt.grid(alpha=0.25)
    plt.xlabel('Year')
    plt.ylabel('Shoreline Position (m)')
    plt.title('Shoreline Position from 2023 Reference Line')
    plt.legend()
    plt.tight_layout()
    plt.close()

    logger.info(f"=== Time series plot succesfully created ===")
    return fig


def PostProcGCTR(df,filtered_df,selected_transect=None):
    """Post-processing GCTR data into 3 plots:
    1. Time series of selected transect compared to the neighbor
    2. Time stack of shoreline position
    3. Time stact of shoreline change

    Args:      
        df: dataframe of GCTR
        filtered_df: dataframe with the transect name (and new ambient rate)

        optional
        transect_name: name of the selected transect to highlight the transect position

    Returns:
        plot: list of plot object
    """ 
    # filter GCTR dataframe with transect data that available in shoreline series
    transect_id = filtered_df['transect_id'].values

    # apply a new rate (if applicable)
    if 'sds_change_rate' in filtered_df.columns:
        print('----------Applying new rate')
        df['sds_change_rate'] = df['transect_id'].map(filtered_df.set_index('transect_id')['sds_change_rate'])
    
    reduced_transect = df[df['transect_id'].isin(transect_id)].dropna(subset='sds_change_rate')

    # call to plotting function
    logger.info(f"=== Creating ambient rate map ===")
    plot1 = PlotCoastlineRate(reduced_transect,selected_transect)
    logger.info(f"=== Creating bar plot ===")
    plot2 = PlotBarCoastalChange(reduced_transect,selected_transect)
    logger.info(f"=== Creating table summary ===")
    plot3 = PlotTable(reduced_transect)

    plot = [plot1,plot2,plot3]

    return plot


def PlotCoastlineRate(df,selected_transect=None):
    """Function for plotting ambient rate on map tiles

    Args:      
        df: dataframe of GCTR
        selected_transect: name of the selected transect to highlight the transect position

        optional
        transect_name: name of the selected transect to highlight the transect position

    Returns:
        plot: plot object
    """     
    # Arrow shaft (from (0,0) to (2,2))
    def plot_arrow(x_ori,y_ori,theta,shaft_length=350,arrow_length=75,
                   reverse=False,color = 'yellow',label = None):

        """Function for drawing arrow on map tiles

        Args:      
            x_ori: x coordinate of arrow origin point
            y_ori: y coordinate of arrow origin point
            theta: direction of the arrow

            optional
            shaft_length: length of arrow shaft in meter
            arrow_length: length of arrow head in meter
            reverse: reverse origin point as end point
            color: arrow and annotation color
            label: text annotation

        Returns:
            hv.plot object: arrow object
        """ 

        # draw arrow shaft
        theta_rad = np.deg2rad(90 - theta)
        x_end = x_ori + shaft_length * np.cos(theta_rad)
        y_end = y_ori + shaft_length * np.sin(theta_rad)
        shaft = hv.Segments([(x_ori, y_ori, x_end, y_end)])

        # draw arrow head
        theta_left = theta-(-150)
        theta_right = theta-150

        # make arrow based on end point
        if reverse:
            x_temp = x_ori
            y_temp = y_ori
            x_ori = x_end
            y_ori = y_end
            x_end = x_temp
            y_end = y_temp
            theta_left = (theta_left + 180)%360
            theta_right = (theta_right + 180)%360
        
        theta_arrow1 = np.deg2rad(90 - theta_left)
        theta_arrow2 = np.deg2rad(90 - theta_right)
        arrow1_end_x = x_end + arrow_length * np.cos(theta_arrow1)
        arrow1_end_y = y_end + arrow_length * np.sin(theta_arrow1)
        arrow2_end_x = x_end + arrow_length * np.cos(theta_arrow2)
        arrow2_end_y = y_end + arrow_length * np.sin(theta_arrow2)

        arrowhead = hv.Segments([
            (x_end, y_end, arrow1_end_x, arrow1_end_y),  # left side
            (x_end, y_end, arrow2_end_x, arrow2_end_y)   # right side
        ])


        if label:
            # define label positon
            if theta>=45 and theta <135:
                ta = 'left'
                tb = 'middle'
            elif theta >= 135 and theta < 225:
                ta = 'center'
                tb = 'top'
            elif theta >= 225 and theta < 315:
                ta = 'right'
                tb = 'middle'
            else:
                ta = 'center'
                tb = 'bottom'

            # draw label
            text = hv.Labels(
                {'x' : x_ori, 
                 'y' : y_ori, 
                 'text' : label},
                 kdims=['x', 'y'], vdims=['text']).opts(
                text_color=color,
                text_font_size="18pt",
                text_baseline= tb,
                text_align = ta
            )
            
            return (shaft*arrowhead*text).opts(
            hv.opts.Segments(color=color, line_width=4)
            )

        return (shaft*arrowhead).opts(
            hv.opts.Segments(color=color, line_width=4)
            )

    # reference length for drawing arrow
    reference_length = 15000


    # make varying transect from polar coordinate
    origin = df.to_crs(epsg=3857).centroid     # convert geometry to web mercator
    scalefac = reference_length/10 / df['sds_change_rate'].abs().max()
    r = df['sds_change_rate'] * scalefac              
    theta_rad = np.deg2rad(90 - df['bearing'])

    x = origin.x + r * np.cos(theta_rad)
    y = origin.y + r * np.sin(theta_rad)

    df['end_point']=[Point(xi, yi) for xi, yi in zip(x, y)]
    end_point = gpd.GeoSeries(df['end_point'])

    # make geo data frame for scaled data with centroid as origin
    line=[]
    for awal, akhir in zip(origin,end_point):
        line.append(LineString([awal,akhir]))
    line_gdf = df['sds_change_rate'].copy()
    line_gdf = gpd.GeoDataFrame(line_gdf).set_geometry(line).set_crs(origin.crs).to_crs(df.crs)
    line_gdf = line_gdf.to_crs(4326)


    # plot limit
    maxc = df['sds_change_rate'].max()
    minc = df['sds_change_rate'].min()
    clim = max(maxc,abs(minc))
    cmap = cc.m_diverging_gwr_55_95_c38_r.copy()
    cmap.set_bad(color="black")

    # draw ambient change map
    plot = line_gdf[['geometry','sds_change_rate']].hvplot(
        geo=True,
        tiles="ESRI",
        color='sds_change_rate',
        frame_width=550,
        frame_height=450,
        colorbar=True,
        cmap=cmap,
        clim=(-clim, clim),
        line_width = 12,
        clabel='Shoreline Change Rate (m/year)'
    )

    # find the first transect as reference point for placing legend
    sorted = df[['transect_id']].sort_values(by=['transect_id'],ascending=False)
    first_transect_name = sorted.values[0]
    index_ref = df['transect_id'].isin(first_transect_name)
    ref_point = df[index_ref].to_crs(epsg=3857).geometry.interpolate(0)

    # draw arrow pointing view orientation
    x_ori = ref_point.x.to_numpy()
    y_ori = ref_point.y.to_numpy()
    shaft_length = reference_length/8
    arrow_length = shaft_length/5
    arrow_x_axis = plot_arrow(x_ori,y_ori,df['bearing'].median()+90,shaft_length,arrow_length,color='yellow')

    # draw arrow orientation anchor
    circle = ref_point.buffer(arrow_length/2)   # radius in CRS units
    circle = gpd.GeoDataFrame(geometry=circle).to_crs(epsg=4326)
    annotations = circle.hvplot(geo=True, 
                                color="yellow", 
                                line_color = 'yellow',
                                alpha=1)

    # draw arrow pointing selected transect
    transect_loc = df[df['transect_id'] == selected_transect]
    xs0 = transect_loc['end_point'].values[0].x
    ys0 = transect_loc['end_point'].values[0].y

    # make arrow lighter or darker based on its position on map
    if transect_loc['sds_change_rate'].values > 0:
        direction = transect_loc['bearing'].values[0]
        color = 'white'
    else:
        direction = (transect_loc['bearing'].values[0]+180)%360
        color ='black'

    transect_pos = plot_arrow(xs0,ys0,direction,shaft_length,arrow_length,
                              color=color,reverse=True,label='Selected Transect')


    result = (plot*annotations*arrow_x_axis*transect_pos)


    logger.info(f"=== Ambient rate map sucessfully created ===")
    return result



def PlotBarCoastalChange(df,selected_transect=None,plot_outlier=False):
    """Bar plot of coastal typology

    Args:      
        df: dataframe of GCTR
        selected_transect: name of the selected transect to highlight the transect position

        optional
        transect_name: name of the selected transect to highlight the transect position
        plot_outlier: highlight outlier in the figure based on local MAD and isolation forest

    Returns:
        plot: plot object
    """ 

    def color_palletes():
        """
            Function for defining color pallete used
        """ 
        # Define shore type colors map
        shore_colors = {
            'muddy_sediments': ('#8B4513', 'Muddy Shoreline'),     # SaddleBrown
            'sandy_gravel_or_small_boulder_sediments': ('#FFD700', 'Sandy Shoreline'),  # Bright Yellow
            'no_sediment_or_shore_platform': ('#D3D3D3', 'No Sediment'),  # Light Gray
            'rocky_shore_platform_or_large_boulders': ('#2F4F4F', 'Rocky Shoreline'),  # Dark Slate Gray
            'None': ("#000000", 'No Data')  # Black
        }

        # Define coastal type color map
        coastal_colors = {
            "bedrock_plain": ("#8b5a2b",'Bedrock Plain'),
            "cliffed_or_steep": ("#654321",'Cliffed or Steep'),
            "dune": ("#e4c580",'Dune'),
            "engineered_structures": ("#7f7f7f",'Engineered Structures'),
            "inlet": ("#1f77b4",'Inlet'),
            "moderately_sloped": ("#a67c52",'Moderately Sloped'),
            "sediment_plain": ("#d49f50",'Sediment Plain'),
            "wetland": ("#3aaf55",'Wetland'),
            'None': ("#000000", 'No Data')  # Black
        }

        return shore_colors, coastal_colors
    

    def local_modified_zscore_outlier(y, k=3, thresh=3.5):
        """
        Local Modified Z-score based outlier detection for 1D data.
        
       Args:
        y : series of ambient change along transect
        k : number of neighbors on each side for local window    
        thresh : modified Z-score threshold (default 3.5)
            
            
        Returns:
        outlier : boolean mask of detected local outliers

        """
        y = np.asarray(y)
        n = len(y)
        outlier = np.zeros(n, dtype=bool)
        z_scores = np.zeros(n)

        for i in range(n):
            start, end = max(0, i - k), min(n, i + k + 1)
            local = y[start:end]
            med = np.median(local)
            mad = median_abs_deviation(local, scale='normal')  # unscaled MAD

            if mad == 0:
                z = 0
            else:
                # Modified Z-score formula
                z = 0.6745 * abs(y[i] - med) / mad
                z = 1 * abs(y[i] - med) / mad


            z_scores[i] = z
            outlier[i] = z > thresh

        return outlier

    def isolation_forest(y,c = 0.005,random_state = 42):
        """
        Outlier detection for based on isolation forest algorithm.
        
       Args:
        y : series of ambient change along transect
        c : contamination parameter   
        random_state : random seed for algorithm
            
            
        Returns:
        outlier : boolean mask of detected local outliers
        """
        X = y.to_numpy().reshape(-1, 1)
        iso = IsolationForest(contamination=c, random_state=random_state)
        iso_labels = iso.fit_predict(X)
        iso_mask = iso_labels == -1
        return iso_mask
    
    # call color pallter
    shore_colors, coastal_colors = color_palletes()

    # sort data based on transect id to make them sequential
    sorted_df = df[['transect_id','geom','sds_change_rate',
                    'class_shore_type','class_coastal_type']].sort_values(by=['transect_id'],ascending=False)
    sorted_df['dist'] = np.arange(0,len(sorted_df)*100,100)

    # Create figure
    fig = plt.figure(figsize=(10,7))

    # Prepare subplots using GridSpec
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1) 
    bar_width = 100
    fig.supxlabel('Distance along Coastline (m)',fontsize=12, y=0.03)

    # Plot Bar graph of coastline change rate
    ax0 = fig.add_subplot(gs[0])
    colors = ['green' if val >= 0 else 'red' for val in sorted_df['sds_change_rate']]
    ax0.bar(sorted_df['dist'],sorted_df['sds_change_rate'],
            color=colors,width=bar_width,align='edge')
    ax0.set_xlim(min(sorted_df['dist']),max(sorted_df['dist']+100))
    ax0.set_ylabel('Coastline Change Rate (m/year)')
    ymin = min(1.6*min(sorted_df['sds_change_rate']),-7)
    ymax = max(1.6*max(sorted_df['sds_change_rate']),7)
    ax0.set_ylim(ymin,ymax)
    ax0.grid(alpha=0.6)


    # Plot arrow pointing the position and make the annotation
    transect_loc = sorted_df[sorted_df['transect_id'] == selected_transect]
    xa = transect_loc['dist'].values[0] +50
    ya = transect_loc['sds_change_rate'].values[0]
    xa_text = xa 
    ya_text = 1.5 * ya + 5 * np.sign(ya)
    ax0.annotate(
        'Selected Transect',
        xy=(xa, ya), xytext=(xa_text, ya_text),
        arrowprops=dict(facecolor='black', shrink=0.05),
        ha='center'
        )


    # plot flagged outlier
    if plot_outlier:
        mad_mask = local_modified_zscore_outlier(sorted_df['sds_change_rate'], k=5, thresh=2.5)
        iso_mask = isolation_forest(sorted_df['sds_change_rate'],c = 0.02,random_state = 42)
        ax0.scatter(sorted_df['dist'][mad_mask], sorted_df['sds_change_rate'][mad_mask], 
                    color='orange', label='MAD outlier', s=60)
        ax0.scatter(sorted_df['dist'][iso_mask], sorted_df['sds_change_rate'][iso_mask], 
                    color='black', label='IF outlier', s=60)

    # Plot Stacked Bar for coastal type and shore type
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    colors = [coastal_colors[f][0] for f in sorted_df['class_coastal_type']]
    ax1.bar(sorted_df['dist'], height=0.5, bottom=0, width=bar_width, color=colors, align='edge',
            edgecolor=None)

    colors = [shore_colors[f][0] for f in sorted_df['class_shore_type']]
    ax1.bar(sorted_df['dist'], height=0.5, bottom=0.5, width=bar_width, color=colors, align='edge',
              edgecolor=None)


    # Set labels 
    ax1.set_ylim([0,1])
    ax1.set_yticks([0.25, 0.75])  # positions
    ax1.set_yticklabels(['Coastal\nType\n\n', 'Shore\nType\n\n'],rotation=90,va='center',ha='center')  # new labels


    # Set legend for coastal/shore type available in bounding box
    handles = []
    used_shore_colors = {k: v for k, v in shore_colors.items() if k in sorted_df['class_shore_type'].unique()}
    used_coastal_colors = {k: v for k, v in coastal_colors.items() if k in sorted_df['class_coastal_type'].unique()}

    # Legend set for Shore Type
    handles.append(mpatches.Patch(color='none', label="Shore Type"))  
    for num,(key, values) in enumerate(used_shore_colors.items()):
        handles.append(mpatches.Patch(color=values[0], label=values[1]))
    for i in range(len(shore_colors)-1-num):
        handles.append(mpatches.Patch(color='none', label=" "))

    # Legend set for Coastal Type
    handles.append(mpatches.Patch(color='none', label="Coastal Type"))
    for num, (key, values) in enumerate(used_coastal_colors.items()):
        handles.append(mpatches.Patch(color=values[0], label=values[1]))
    for i in range(len(coastal_colors)-1-num):
        handles.append(mpatches.Patch(color='none', label=" "))

    # Plot legend to figure
    legend = plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.5),
                   ncol=3, frameon=True, handlelength=1)
    
    for text, handle in zip(legend.get_texts(), legend.legend_handles):
        if handle.get_facecolor()[0] == 0 and handle.get_facecolor()[3] == 0: 
            text.set_weight("bold")
    plt.close()

    logger.info(f"=== Bar plot succesfully created ===")
    return fig


def PlotTable(df):
    """Calculation of area statistic

    Args:      
        df: dataframe of GCTR data

    Returns:
        styled: styled table object
    """ 
    # create and calculate the table entry
    table = df[['sds_change_rate']].describe().round(2)
    table = (table.drop(table.index[4:7]) )
    table = table.rename(columns={'sds_change_rate': 'Value'})
    table = table.rename(index={'count': 'length'})
    table = table.rename(index={'mean': 'mean rate'})
    table = table.rename(index={'max': 'max rate'})
    table = table.rename(index={'min': 'min rate'})
    table.loc['length'] = table.loc['length'] * 100
    erosion_idx = df['sds_change_rate']<0
    erosion = df['sds_change_rate'][erosion_idx]
    sedimentation = df['sds_change_rate'][~erosion_idx]
    table.loc['erosion'] = erosion.sum() * 100
    table.loc['sedimentation'] = sedimentation.sum() * 100
    table.loc['net budget'] = table.loc['sedimentation'] + table.loc['erosion']
    table['Unit'] = ['m','m/year','m/year','m/year','m/year','m2/year','m2/year','m2/year']

    # Table styling
    styled = table.style.set_caption("Summary Statistics") \
                    .format({'Value': "{:.2f}", 'Unit': "{:s}"}) \
                    .set_table_styles([
                        {"selector": "caption",
                            "props": [("caption-side", "top"), ("font-size", "16px"), ("font-weight", "bold")]}
                    ])

    logger.info(f"=== Summary table succesfully created ===")
    return styled
