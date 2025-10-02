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
from dotenv import load_dotenv
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import logging
logger = logging.getLogger("PYWPS")

# some generic to load env file
load_dotenv()
logger.info('env settings read')

abspath = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    abspath = 'C:/develop/shoreline-monitor-wps/data/'
    location = 'localhost:5000'
else:
    abspath = 'opt/pywps/data/'
    location = 'https://shoreline-monitor.avi.directory.intra/wps'

logger.info('abspath set',abspath)
logger.info('location set',location)

# PostgreSQL connection (adjust as needed)
pg_user = os.getenv("PG_USER")
pg_pass = os.getenv("PG_PASS")
pg_host = os.getenv("PG_HOST")
logger.info('host/username',pg_host,pg_user)
pg_host = 'c-oet30001.directory.intra'
pg_db = os.getenv("PG_DB")
pg_port = 5432

engine = create_engine(
    f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
)
try:
    engine.connect()
    logger.info("successfull connection to database")
except SQLAlchemyError as err:
    logger.info("error connecting", err.__cause__)

def scatterplot(df,dfm):
    """_summary_

    Args:
        df (dataframe): data of the transect (derive from shoreline monitor series data)
        dfm (dataframe): metadata of the transect (gctr data)

    Returns:
        html: html document with scatterplot, trenline and metadata of the transect
    """
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

    print('regression line calculated')

    fig.add_trace(go.Scatter(
    x=df['datetime'].iloc[0] + (df['datetime'].iloc[-1] - df['datetime'].iloc[0]) * np.linspace(0, 1, 100),
    y=y_pred,
    mode='lines',
    type='scatter',
    line=dict(color='red'),
    name='Regression Line'
    ))

    print('scatter and regression line created')

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
    htmlfile = os.path.join(abspath,pltname)
    logger.info('htmlfile',htmlfile)
    fig.write_html(htmlfile,auto_play=False)

    logger.info('plot created')

    assignmetadata(htmlfile,dfm)
    logger.info('metadata added to plot')
    
    # based on the pltname, define the url needed to pass to frontend
    url = f'{location}/data/{pltname}'
    return url

def assignmetadata(html,dfm):
    """This function adds metadata to the scatterplot created in the previous routine.
    This metadata gives overal description of the transect.

    Args:
        html (string): html file with scatterplot
        dfm (dataframe): dataframe with basic metadata derived from gctr table

    Returns:
        html (string): overwrites the created html file by appending a specific portion of html code and contents
    """
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
    #newfile = html.replace('.html','_m.html')
    with open(html, 'w', encoding='utf-8') as f:
        f.write(plot_html + html_info)
    
    return html

def handler(profile):
    """This function is the main handler that recieves data from the WPS service.
        The function constructs an HTML page with information from the profile (metadata called) 
        and data from the associated points in time and returns an html page with both data integrated

        Args:
            profile (integer): profile id of the transect/profile, bear in mind, 
            this is not the transect_id, but a unique index generated after reading the data

        Returns:
            html (string): link to an html file with information on the profile (transect) and 
                           timeseries data.
    """    
    """
    first part selects the metadata
    """
    logger.info('handler, profile',profile)
    
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
    
    df = pd.read_sql_query(strsql,engine)
    
    #scond part gets the profile data
    strsql = f"""select datetime, shoreline_position 
                 from shorelinemonitor_series 
                 where gctr_id = '{profile}'
                 order by datetime"""
    dfp = pd.read_sql_query(strsql,engine)

    # okay, there is an issue here, there are shorelines without points!
    # to be exact, some figures:
    # - gctr has 11545312 unique transect_id
    # - shoreline_monitor_series has 7471154 unique transect ids 
    # DIFFERENCE is 3.5 * 10^6 ... records, meaning, for 3.5 miljon profiles, there is no data available
    # need to do something with that
    
    if len(dfp) == 0:
        logger.info('no derived measurements available for id', profile)
 
    url = scatterplot(dfp,df)
 
    return url
    
def test():
    print(handler(8918455))