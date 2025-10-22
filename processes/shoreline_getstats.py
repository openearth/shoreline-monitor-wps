#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2025 Deltares
#     Ioanna Micha
#     ioanna.micha@deltares.nl
#     Gerrit Hendriksen
#     gerrit.hendriksen@deltares.nl
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

import os
import time
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import configparser
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import logging
from .generate_plot import *

logger = logging.getLogger("PYWPS")

# Global variables to cache configuration
_config_initialized = False
_engine = None
_abspath = None
_location = None


def _initialize_config():
    """Initialize configuration - runs once per request when PyWPS is ready"""
    global _config_initialized, _engine, _abspath, _location

    if _config_initialized:
        return  # Already initialized

    # Load configuration file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_file = os.path.join(project_root, "configuration.txt")

    config = configparser.ConfigParser()
    config.read(config_file, encoding="utf-8")

    # Set paths - use relative paths from project root
    if os.name == "nt":
        _abspath = os.path.join(project_root, "data")
        _location = "localhost:5000"

    else:
        _abspath = os.path.join(project_root, "data")
        _location = "https://shoreline-monitor.openearth.eu"

    # PostgreSQL connection - read from config file
    pg_user = config.get("database", "PG_USER")
    pg_pass = config.get("database", "PG_PASS")
    pg_host = config.get("database", "PG_HOST")
    pg_db = config.get("database", "PG_DB")
    pg_port = 5432

    try:
        _engine = create_engine(
            f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        )
        # Test connection
        _engine.connect().close()
        logger.info("Database connection successful")
    except SQLAlchemyError as err:
        logger.error(f"Database connection failed: {err}")
        raise
    except Exception as err:
        logger.error(f"Unexpected error creating database connection: {err}")
        raise

    _config_initialized = True

def handler(profile):
    """This function is the main handler that receives data from the WPS service.
    The function constructs an HTML page with information from the profile (metadata called)
    and data from the associated points in time and returns an html page with both data integrated

    Args:
        profile (integer): profile is equal to index in the gctr table, bear in mind,
        this is not the transect_id, but a unique index generated after reading/importing the data

    Returns:
        html (string): link to an html file with information on the profile (transect) and
                       timeseries data.
    """
    global _abspath, _location
    
    logger.info(f"=== Starting handler for profile: {profile} ===")

    # Initialize configuration (this will now run when PyWPS is ready)
    _initialize_config()
 
    logger.info("Executing metadata query")
    strsql = f"""
        WITH target_profile AS (
            SELECT geom_3857, index
            FROM gctr
            WHERE index = {profile}
        )
        SELECT 
            g.geom_3857 as geom,
            degrees(ST_Azimuth(st_startpoint(g.geometry), st_endpoint(g.geometry))) as bearing,
            g.transect_id,
            g.sds_change_rate,
            g.class_shore_type,
            g.class_coastal_type 
        FROM gctr g
        CROSS JOIN target_profile t
        WHERE st_dwithin(g.geom_3857, t.geom_3857, 5000)
    """

    df = gpd.read_postgis(strsql, _engine)

    if os.name =='nt':
        print(f"Metadata query returned {len(df)} rows")
    else:
        logger.info(f"Metadata query returned {len(df)} rows")
    

    logger.info("Executing profile data query")
    # create a list of transect_ids in order to select all corresponding 
    # shoreline-monitor-series data in following query
    lsttransects = ','.join(f"'{tid}'" for tid in df['transect_id'])
    strsql = f"""select transect_id, 
                    st_transform(geometry,3857) as geom, 
                    datetime, 
                    shoreline_position, 
                    obs_is_primary
                 from shorelinemonitor_series 
                 where transect_id in ({lsttransects})
                 order by datetime"""
    
    dfp = gpd.read_postgis(strsql, _engine)

    if os.name == 'nt':
        print(f"Profile data query returned {len(dfp)} rows")
    else:
        logger.info(f"Profile data query returned {len(dfp)} rows")

    # Handle case where no data is available
    if len(dfp) == 0:
        logger.warning(f"No derived measurements available for profile: {profile}")
        return None  # or return an error message

    logger.info("Generating 6 plots")


    def find_transect_name(profile):
        """Function for finding transect name from profile id

        Args:
            profile: profile number in integer

        Returns:
            string: name of corresponding transect
        """ 

    
        strsql = f"""SELECT 
                transect_id,
                sds_change_rate,
                class_shore_type,
                class_coastal_type 
                from gctr
                where index = '{profile}'"""

        df = pd.read_sql_query(strsql, _engine)
        return df['transect_id'].values[0]

    
    transect_name = find_transect_name(profile)
    filter, plot = PostProcShoreline(transect_name,dfp)
    plot_GCTR = PostProcGCTR(df,filter,selected_transect=transect_name)

    # Create lightweight HTML with minimal JavaScript
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    
    # Create a single dashboard with all plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Series', 'Shoreline Position Heatmap',
                       'Annual Change Heatmap', 'Coastal Typology'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add time series plot
    time_series_fig = plot[0]
    for trace in time_series_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Add heatmap plots
    heatmap_pos_fig = plot[1]
    for trace in heatmap_pos_fig.data:
        fig.add_trace(trace, row=1, col=2)
    
    heatmap_diff_fig = plot[2]
    for trace in heatmap_diff_fig.data:
        fig.add_trace(trace, row=2, col=1)
    
    # Add bar chart
    bar_fig = plot_GCTR[1]
    for trace in bar_fig.data:
        fig.add_trace(trace, row=2, col=2)
    
    # Optimize layout for speed
    fig.update_layout(
        title=f'Shoreline Analysis - Profile {profile}',
        height=800,
        width=1200,
        showlegend=True,
        template='plotly_white',
        modebar_remove=['pan2d', 'lasso2d', 'select2d', 'autoScale2d', 'resetScale2d']
    )

    # define unique id based on current time
    id = str(int(time.time() * 1000))
    pltname = f"stat_{id}.html"

    # define htmlfile to write to serverside place and store
    htmlfile = os.path.join(_abspath, pltname)
    logger.info(f"Writing optimized plot to: {htmlfile}")
    
    # Save with minimal configuration for fastest loading
    pyo.plot(fig, filename=htmlfile, auto_open=False, 
             config={
                 'displayModeBar': True, 
                 'displaylogo': False,
                 'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
                 'responsive': True
             },
             include_plotlyjs='cdn')

    logger.info("Plot file created")

    # the url below is the URL that needs to be return to WPS process
    url = f"{_location}/data/{pltname}"
    logger.info(f"=== Handler completed successfully, returning URL: {url} ===")
    return url

def test():
    """Test function"""
    print(
        handler(2015066)
    )  # 2805066 #"cl30793s01tr02935165" #"cl33097s00tr00002666" 2805066


if __name__ == "__main__":
    test()
