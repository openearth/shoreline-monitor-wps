#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2025 Deltares
#     Ioanna Micha
#     ioanna.micha@deltares.nl
#     Gerrit Hendriksen
#     gerrit.hendriksen@deltares.nl	

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
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import configparser
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import logging

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
    config_file = os.path.join(project_root, 'configuration.txt')
    
   
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    
    # Set paths - use relative paths from project root
    if os.name == 'nt':
        _abspath = os.path.join(project_root, 'data')
        _location = 'localhost:5000'


    else:
        _abspath = os.path.join(project_root, 'data')
        _location = 'https://shoreline-monitor.avi.directory.intra'
    

    
    # PostgreSQL connection - read from config file
    pg_user = config.get('database', 'PG_USER')
    pg_pass = config.get('database', 'PG_PASS')
    pg_host = config.get('database', 'PG_HOST')  
    pg_db = config.get('database', 'PG_DB')
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

def scatterplot(df, dfm):
    """Create scatterplot with regression line and metadata

    Args:
        df (dataframe): data of the transect (derive from shoreline monitor series data)
        dfm (dataframe): metadata of the transect (gctr data)

    Returns:
        html: html document with scatterplot, trendline and metadata of the transect
    """
    global _abspath, _location
    
    logger.info("Starting scatterplot generation")
    
    # Create basic scatter plot
    fig = go.Figure(data=[go.Scatter(
        x=df['datetime'],
        y=df['shoreline_position'],
        mode='markers',
        type='scatter',
        name='Shoreline Position'
    )])

    # Calculate regression line
    X = np.array(df.index).reshape(-1, 1)
    y = df['shoreline_position'].values
    model = LinearRegression().fit(X, y)
    x_pred = np.linspace(0, len(df) - 1, 100).reshape(-1, 1)
    y_pred = model.predict(x_pred)

    logger.info('Regression line calculated')

    fig.add_trace(go.Scatter(
    x=df['datetime'].iloc[0] + (df['datetime'].iloc[-1] - df['datetime'].iloc[0]) * np.linspace(0, 1, 100),
    y=y_pred,
    mode='lines',
    type='scatter',
    line=dict(color='red'),
    name='Regression Line'
    ))

    logger.info('Scatter and regression line created')

    # Update layout
    fig.update_layout(
        title='Shoreline Position Over Time',
        xaxis_title='Date',
        yaxis_title='Shoreline Position'
    )

    # define unique id based on current time
    id = str(int(time.time() * 1000))
    pltname = f'plot_{id}.html'

    # define htmlfile to write to serverside place and store
    htmlfile = os.path.join(_abspath, pltname)
    logger.info(f'Writing plot to: {htmlfile}')
    fig.write_html(htmlfile, auto_play=False)

    logger.info('Plot file created')

    assignmetadata(htmlfile, dfm)
    logger.info('Metadata added to plot')
    
    # based on the pltname, define the url needed to pass to frontend
    url = f'{_location}/data/{pltname}'
    logger.info(f'Generated URL: {url}')
    return url

def assignmetadata(html, dfm):
    """This function adds metadata to the scatterplot created in the previous routine.
    This metadata gives overall description of the transect.

    Args:


         html (string): html file with scatterplot
         dfm (dataframe): dataframe with basic metadata derived from gctr table

    Returns:
         html (string): overwrites the created html file by appending a specific portion of html code and contents
    """
    logger.info("Adding metadata to plot")
    
    # Create a HTML string with information
    html_info = '''
    <div>
        <h2>Plot Information</h2>
        <table>
            <tr>
                <th>Country</th>
                <th>Continent</th>
                <th>Sandy</th>
                <th>Ambient Change Rate</th>
            </tr>
    '''

    for index, row in dfm.iterrows():
        html_info += f'''
            <tr>
                <td>{row["country"]}</td>
                <td>{row["continent"]}</td>
                <td>{str(row["sandy"])}</td>
                <td>{row["sds_change_rate"]}</td>
            </tr>
        '''

    html_info += '''
        </table>
        </div>
        '''
    
    # Read the plot HTML file
    with open(html, 'r', encoding='utf-8', errors='replace') as f:
        plot_html = f.read()

    # Add the information section to the plot HTML
    with open(html, 'w', encoding='utf-8') as f:
        f.write(plot_html + html_info)
    
    logger.info("Metadata successfully added")
    return html

def handler(profile):
    """This function is the main handler that receives data from the WPS service.
        The function constructs an HTML page with information from the profile (metadata called) 
        and data from the associated points in time and returns an html page with both data integrated

        Args:
            profile (integer): profile id of the transect/profile, bear in mind, 
            this is not the transect_id, but a unique index generated after reading the data

        Returns:
            html (string): link to an html file with information on the profile (transect) and 
                           timeseries data.
    """    
    logger.info(f"=== Starting handler for profile: {profile} ===")
    
    # Initialize configuration (this will now run when PyWPS is ready)
    _initialize_config()
    
    logger.info("Executing metadata query")
    strsql = f"""SELECT 
                g.transect_id as transect_id,
                common_country_name as country,
                continent,
                sandy,
                sds_change_rate,
                index
                FROM public.gctr g
                join country c on c.idccn = g.idccn
                join continent ct on ct.idcnt = g.idcnt
                where g.index = {profile}"""
    
    df = pd.read_sql_query(strsql, _engine)
    logger.info(f"Metadata query returned {len(df)} rows")
    
    logger.info("Executing profile data query")
    strsql = f"""select datetime, shoreline_position 
                 from shorelinemonitor_series 
                 where gctr_id = '{profile}'
                 order by datetime"""
    dfp = pd.read_sql_query(strsql, _engine)
    logger.info(f"Profile data query returned {len(dfp)} rows")

    # Handle case where no data is available
    if len(dfp) == 0:
        logger.warning(f'No derived measurements available for profile: {profile}')
        return None  # or return an error message
 
    logger.info("Generating scatterplot")
    url = scatterplot(dfp, df)
    logger.info(f"=== Handler completed successfully, returning URL: {url} ===")
    return url
    
def test():
    """Test function"""
    print(handler(2805066))

if __name__ == "__main__":
    test()