import matplotlib as mpl
from shapely.geometry import Point, LineString
from sklearn.linear_model import LinearRegression
import geopandas as gpd
import holoviews as hv
import hvplot.pandas
import colorcet as cc
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
import panel as pn
import logging

hv.extension("bokeh")
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
    
    # filtering the dataframe with primary observation~
    mask = df['obs_is_primary'] & ~df['obs_is_outlier']
    df_filtered = df[mask]

    # validating date input
    if not start and not end:
        date_filter = df_filtered
    else:
        date_filter = df_filtered[
            (df_filtered["datetime"] >= start if start is not None else True) &
            (df_filtered["datetime"] <= end if end is not None else True)
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
    # selected_transect = pivot.index.get_loc(transect_name)

    # replace transect information to the relative distance from the first data
    dist = np.arange(0,len(pivot)*100,100)
    pivot.index = dist

    # call to each plotting function
    logger.info(f"=== Creating time series plot ===")
    plot1 = PlotSelectedTimeSeries(df_filtered,transect_name)
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

    lower_quantile = pivot.min().quantile(0.025)
    upper_quantile = pivot.max().quantile(0.975)

    # Set the plot range considering the quantiles
    min_range = min(lower_quantile, -15)
    max_range = max(upper_quantile, 15)

    # Ensure that the color limits are symmetric around zero
    max_abs_range = max(abs(min_range), abs(max_range))
    clim = (-max_abs_range, max_abs_range)

    # Retransform table to long dataframe
    df_long = pivot.reset_index().melt(id_vars='index',var_name='Year', value_name='Shoreline Position (m)')
    df_long.rename(columns={'index':'Distance (m)'},inplace=True)

    cmap = mpl.colormaps['RdYlGn'].reversed()
    fig = df_long.hvplot.heatmap(
        x='Distance (m)',
        y='Year',
        C='Shoreline Position (m)',
        cmap=cmap,
        colorbar=True,
        height=400,
        width=500,
        clim=clim,
        title="Shoreline Position from 2023 Reference Line",
        clabel = 'Shoreline Position (m)'
    ).opts(
        active_tools=[],
    )

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

    # Retransform table to long dataframe
    df_long = pivot_change.reset_index().melt(id_vars='index',var_name='Year', value_name='Change rate (m/year)')
    df_long.rename(columns={'index':'Distance (m)'},inplace=True)


    # cmap = mpl.colormaps['RdYlGn'].reversed()
    cmap = mpl.colormaps['RdYlGn']
    fig = df_long.hvplot.heatmap(
        x='Distance (m)',
        y='Year',
        C='Change rate (m/year)',
        cmap=cmap,
        colorbar=True,
        height=400,
        width=500,
        clim=clim,
        title="Yearly Shoreline Change",
        clabel = 'Shoreline Change (m/year)'
    ).opts(
        active_tools=[],
    )

    logger.info(f"=== Time stack plot of annual shoreline change succesfully created ===")
    return fig


def PlotSelectedTimeSeries(df,selected_transect):
    """Plotting table of spatio-temporal annual shoreline change:

    Args:      
        df: dataframe of shoreline position that have been filtered
        selected_transect: name of the selected transect

    Returns:
        plot: plot object
    """ 
    
    pivot_ts = df.pivot_table(
        # index='transect_id', 
        # columns='datetime', 
        index='datetime', 
        columns='transect_id', 
        values='shoreline_position', 
        aggfunc='mean'  # or sum, max, min, etc.
    )

    dist = np.arange(0,pivot_ts.shape[1]*100,100)
    idx_ts =  pivot_ts.columns == selected_transect
    pivot_ts.columns = dist
    pivot_ts.rename_axis(columns = "Alongshore Distance (m)",inplace=True)

    # Split selected vs others
    df_selected = pivot_ts.loc[:, idx_ts]
    df_others = pivot_ts.loc[:, ~idx_ts]

    # Plot the neighbouring transect
    plot_others = df_others.hvplot.line(
        color='lightgray',
        line_width=1,
        legend =False
    )

    plot_label = df_selected.hvplot.line(
        color='lightgray',
        line_width=1,
        label = 'Neighboring Transect',
    )

    # Highlight selected transect (red, with markers)
    plot_selected = df_selected.hvplot.line(
        color='red',
        line_width=3,
        label = 'Selected Transect',
    )

    # Combine plots
    fig = (plot_others * plot_label * plot_selected).opts(
        height=400,
        width=600,
        title='Shoreline Position from 2023 Reference Line (m)',
        xlabel='Year',
        ylabel='Shoreline Position (m)',
        gridstyle={'alpha': 0.25},
        tools=['hover'],
        active_tools=[],
        legend_position = 'bottom'
    )

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
        plot2 = PlotBarCoastalChange(reduced_transect,selected_transect,plot_outlier=True)
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
    line_gdf = df[['transect_id','sds_change_rate','bearing']].copy()
    # line_gdf = df.copy()
    line_gdf = gpd.GeoDataFrame(line_gdf).set_geometry(line).set_crs(origin.crs).to_crs(df.crs)
    line_gdf = line_gdf.to_crs(4326)

    # find the first transect as reference point for placing legend
    sorted = line_gdf[['transect_id','geometry','sds_change_rate']].sort_values(by=['transect_id'],ascending=False)
    first_transect_name = sorted.values[0]
    index_ref = df['transect_id'].isin(first_transect_name)
    ref_point = df[index_ref].to_crs(epsg=3857).geometry.interpolate(0)
    sorted['dist'] = np.arange(0,len(sorted)*100,100)

    # plot limit
    maxc = df['sds_change_rate'].max()
    minc = df['sds_change_rate'].min()
    clim = max(maxc,abs(minc))
    cmap = mpl.colormaps['RdYlGn']
    # cmap = cc.m_diverging_gwr_55_95_c38_r.copy()
    cmap.set_bad(color="black")

    # draw ambient change map
    plot = sorted[['geometry','sds_change_rate','dist']].hvplot(
        geo=True,
        tiles="ESRI",
        color='sds_change_rate',
        frame_width=600,
        frame_height=450,
        colorbar=True,
        cmap=cmap,
        clim=(-clim, clim),
        line_width = 12,
        title='Shoreline Change Map',
        clabel='Shoreline Change Rate (m/year)',
        hover_cols=['dist'],
        tools=['hover'],
        hover_tooltips=[ 
            ('Distance (m)', '@dist'),
            ('Change rate (m/year)', '@sds_change_rate'),
        ],
    ).opts()

    # # find the first transect as reference point for placing legend
    # sorted = df[['transect_id']].sort_values(by=['transect_id'],ascending=False)
    # first_transect_name = sorted.values[0]
    # index_ref = df['transect_id'].isin(first_transect_name)
    # ref_point = df[index_ref].to_crs(epsg=3857).geometry.interpolate(0)

    # draw arrow pointing view orientation
    x_ori = ref_point.x.to_numpy()
    y_ori = ref_point.y.to_numpy()
    shaft_length = reference_length/8
    arrow_length = shaft_length/5
    # arrow_x_axis = plot_arrow(x_ori,y_ori,df['bearing'].median()+90,shaft_length,arrow_length,color='yellow')
    arrow_x_axis = plot_arrow(x_ori,y_ori,df[index_ref]['bearing'].values+90,shaft_length,arrow_length,color='yellow')


    # draw arrow orientation anchor
    circle = ref_point.buffer(arrow_length/2)   # radius in CRS units
    circle = gpd.GeoDataFrame(geometry=circle).to_crs(epsg=4326)
    annotations = circle.hvplot(geo=True, 
                                color="yellow", 
                                line_color = 'yellow',
                                alpha=1)

    # draw symbol pointing selected transect
    transect_loc = df[df['transect_id'] == selected_transect]
    xs0 = transect_loc['end_point'].values[0].x
    ys0 = transect_loc['end_point'].values[0].y

    transect_pos = hv.Scatter((xs0,ys0),label = 'Selected Transect').opts(color='white', size=20,
         marker='star',line_color = 'black')


    result = (plot*annotations*arrow_x_axis*transect_pos).opts(
        tools=['hover'],
        hover_tooltips=[ 
            ('Change Rate', '@sds_change_rate'),
        ],
        legend_position = 'top'
    )


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

    # create bar plot
    colors = ['green' if val >= 0 else 'red' for val in sorted_df['sds_change_rate']]
    sorted_df['color'] = colors
    bar = sorted_df[['dist','sds_change_rate','color']].hvplot.bar(
        x = 'dist',
        y = 'sds_change_rate',
        color='color',
        width=700, height=400, 
        title='Shoreline Change Rate',
        xlabel='Distance (m)', ylabel='Change Rate',
    ).opts(
        bar_width = 10000,
        hover_tooltips=[ 
            ('Distance (m)', '@dist'),
            ('Change rate (m/year)', '@sds_change_rate'),
        ],
        xlim = (0,sorted_df['dist'].max()),
    )

    # Highlight selected transect
    # selected_transect = 'cl48218s00tr01416537'
    transect_loc = sorted_df[['dist','sds_change_rate']][sorted_df['transect_id'] == selected_transect]
    highlight = transect_loc.hvplot.scatter(
        x='dist', y='sds_change_rate', color='orange', size=100,
        marker='triangle', legend=True,label = 'Selected Transect',
    )

    # Plot outlier
    if plot_outlier:
        mad_mask = local_modified_zscore_outlier(sorted_df['sds_change_rate'], k=5, thresh=2.5)
        iso_mask = isolation_forest(sorted_df['sds_change_rate'],c = 0.02,random_state = 42)
        combined_mask = mad_mask | iso_mask
        outlier = sorted_df[['dist','sds_change_rate']][combined_mask]
        plot_outlier = outlier.hvplot.scatter(
            x='dist', y='sds_change_rate', color='red', size=100,
            marker='circle', legend=True,label = 'Possible Outlier',
        )
        bar_plot = (bar*plot_outlier*highlight).opts(legend_position = 'top')
    else:
        bar_plot = (bar*highlight).opts(legend_position = 'top')


    # Plot Stacked Bar for coastal type and shore type
    map_coastal_color = sorted_df['class_coastal_type'].map(coastal_colors)
    coastal_color_mapped = [t[0] for t in map_coastal_color]
    coast_rename = [t[1] for t in map_coastal_color]
    map_shore_color = sorted_df['class_shore_type'].map(shore_colors)
    shore_color_mapped = [t[0] for t in map_shore_color]
    shore_rename = [t[1] for t in map_shore_color]

    # Prepare additional column for plotting 
    sorted_df['dummy'] = 1
    sorted_df['dummy2'] = -1
    sorted_df['coastal_color'] = coastal_color_mapped
    sorted_df['coast'] = coast_rename
    sorted_df['shore_color'] = shore_color_mapped
    sorted_df['shore'] = shore_rename



    # Coastal type bar
    coast_bar = sorted_df[['dist','dummy2','coastal_color','coast']].hvplot.bar(
        x='dist', y='dummy2', 
        # stacked = True,
        color='coastal_color',
        # label = 'coast',
        # width=700, height=300, 
        xlabel='Distance (m)',
        hover_cols=['coast']
    ).opts(bar_width = 10000,
        tools=['hover'],
        hover_tooltips=[ 
            ('Distance (m)', '@dist'),
            ('Coastal Type', '@coast'),
        ]
    )
    # Shore type bar
    shore_bar = sorted_df[['dist','dummy','shore_color','shore']].hvplot.bar(
        x='dist', y='dummy', 
        # stacked = True,
        color='shore_color',
        # label = 'shore',
        # width=700, height=300, 
        xlabel='Distance (m)',
        hover_cols=['shore']
    ).opts(bar_width = 10000,
        tools=['hover'],
        hover_tooltips=[ 
            ('Distance (m)', '@dist'),
            ('Shore Type', '@shore'),
        ]
    )


    # Combined typology
    typology_plot = (coast_bar * shore_bar).opts(
        yticks=[(-0.5, 'Coastal\nType'), (0.5, 'Shore\nType')],
        # yrotation = 90,
        ylabel = '',
        title = 'Typology',
        show_legend=True,
        legend_position='bottom',
        legend_cols=4,
        width=700, 
        height=150,
        xlim = (0,sorted_df['dist'].max()),    
    )

    # Create legend
    used_shore_colors = {k: v for k, v in shore_colors.items() if k in sorted_df['class_shore_type'].unique()}
    used_coastal_colors = {k: v for k, v in coastal_colors.items() if k in sorted_df['class_coastal_type'].unique()}

    legend_elements = []

    legend_elements.append(hv.Text(x=0, y=1.2, text='Shore Type').opts(
            text_font_size='12pt', text_align='left', text_baseline='middle',
            text_font_style='bold',toolbar=None, default_tools = []))

    # Legend pathces and text for shore type
    for i,(key, values) in enumerate(used_shore_colors.items()):
        # Draw a small rectangle
        legend_elements.append(hv.Rectangles([(0, -i, 0.5, -i+0.4)]).opts(
            color=values[0], line_color='black',
            toolbar=None, default_tools = []
        ))
        # Draw text next to it
        legend_elements.append(hv.Text(x=0.6, y=-i+0.2, text=values[1]).opts(
            text_font_size='10pt', text_align='left', text_baseline='middle',
            toolbar=None, default_tools = []
        ))

    # Legend pathces and text for coastal type
    gap = 6
    nrow = max(len(used_shore_colors),3)

    legend_elements.append(hv.Text(x=0+gap, y=1.2, text='Coastal Type').opts(
            text_font_size='12pt', text_align='left', text_baseline='middle',
            text_font_style='bold',toolbar=None, default_tools = []))

    for i,(key, values) in enumerate(used_coastal_colors.items()):
        
        if i % nrow == 0 and i != 0:
            gap = gap +6

        increment = i % nrow

        # Draw a small rectangle
        legend_elements.append(
            hv.Rectangles([(0+gap, -increment, 0.5 + gap, -increment+0.4)]).opts(
            color=values[0], line_color='black',
            toolbar=None, default_tools = []
        ))
        # Draw text next to it
        legend_elements.append(
            hv.Text(x=0.6 + gap, y=-increment+0.2, text=values[1]).opts(
            text_font_size='10pt', text_align='left', text_baseline='middle',
            toolbar=None, default_tools = []
        ))
        

    # Overlay legend elements
    manual_legend = hv.Overlay(legend_elements).opts(
        width=700, height=150, xaxis=None, yaxis=None, 
        xlim = (0,24),
        ylim = (-i*1.1,2),
    )

    fig = (bar_plot + typology_plot + manual_legend).opts(shared_axes=True).cols(1)


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
