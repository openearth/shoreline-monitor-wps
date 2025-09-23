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
from dotenv import load_dotenv
import plotly.graph_objs as go
# import plotly.offline as pyo
from sklearn.linear_model import LinearRegression
import numpy as np

# some generic to load env file
load_dotenv()

abspath = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    abspath = 'C:/develop/shoreline-monitor-wps/data/'
    location = 'localhost:5000'
else:
    location = 'blabla.avi.directory'

# PostgreSQL connection (adjust as needed)
pg_user = os.getenv("PG_USER")
pg_pass = os.getenv("PG_PASS")
pg_host = os.getenv("PG_HOST")
pg_db = os.getenv("PG_DB")
pg_port = 5432

engine = create_engine(
    f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
)

def scatterplot(df,dfm):
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

    fig.add_trace(go.Scatter(
    x=df['datetime'].iloc[0] + (df['datetime'].iloc[-1] - df['datetime'].iloc[0]) * np.linspace(0, 1, 100),
    y=y_pred,
    mode='lines',
    type='scatter',
    line=dict(color='red'),
    name='Regression Line'
    ))

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
    fig.write_html(htmlfile,auto_play=False)
    assignmetadata(htmlfile,dfm)
    
    # based on the pltname, define the url needed to pass to frontend
    url = f'{location}/data/{pltname}'
    return url

def assignmetadata(html,dfm):
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
    with open(html, 'r') as f:
        plot_html = f.read()

    # Add the information section to the plot HTML
    #newfile = html.replace('.html','_m.html')
    with open(html, 'w') as f:
        f.write(plot_html + html_info)
    return html

def handler(profile):
    """
    first part selects the metadata
    """
    print('profile in handler',profile)
    strsql = f"""SELECT 
                g.transect_id,
                common_country_name as country,
                continent,
                sandy,
                sds_change_rate
                FROM public.gctr g
                join country c on c.idccn = g.idccn
                join continent ct on ct.idcnt = g.idcnt
                where g.index = {profile}""" 
    
    df = pd.read_sql_query(strsql,engine)

    print('profile in handler',profile, len(df))
    #profile = 8918455
    #scond part gets the profile data
    strsql = f"""select datetime, shoreline_position 
                 from shorelinemonitor_series 
                 where gctr_id = '{profile}'
                 order by datetime"""
    dfp = pd.read_sql_query(strsql,engine)
    print(len(dfp))
    url = scatterplot(dfp,df)
    print('url',url)
    return url
    
def test():
    print(handler(8918455))