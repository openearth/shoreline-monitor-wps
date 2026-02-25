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
    return _abspath


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
    df_prim_obs_no_outlier = df[df["obs_is_primary"] & ~df["obs_is_outlier"]]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df_prim_obs_no_outlier["datetime"],
                y=df_prim_obs_no_outlier["shoreline_position"],
                mode="markers",
                type="scatter",
                name="Shoreline Positions",
            )
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=df[df["obs_is_outlier"]]["datetime"],
            y=df[df["obs_is_outlier"]]["shoreline_position"],
            mode="markers",
            type="scatter",
            name="Outlier(s)",
            marker=dict(color="red", symbol="x"),
            visible="legendonly",
        )
    )

    # Option 0: Calculate regression line (wrong, includes all obs)
    # X = np.array(df.index).reshape(-1, 1)
    # y = df["shoreline_position"].values
    # model = LinearRegression().fit(X, y)
    # x_pred = np.linspace(0, len(df) - 1, 100).reshape(-1, 1)
    # y_pred = model.predict(x_pred)

    # logger.info("Regression line calculated")

    # Option 1: Calculate regression line (correct, includes only primary obs and excludes outliers)
    X = (
        df_prim_obs_no_outlier["datetime"].dt.year.values
        - df_prim_obs_no_outlier["datetime"].dt.year.values[0]
    )
    y = df_prim_obs_no_outlier.shoreline_position.values
    model = LinearRegression().fit(X.reshape(-1, 1), y)
    x_values = df_prim_obs_no_outlier["datetime"].dt.year.values
    y_values = model.intercept_ + model.coef_[0] * X

    logger.info("Regression line calculated")

    # Option 2: Use change rate from gctr table, but change_intercept seems to be wrong; need to check with Floris Calkoen..
    # df_prim_obs_no_outlier = df[df["obs_is_primary"] & ~df["obs_is_outlier"]]
    # x_values = df_prim_obs_no_outlier["datetime"].dt.year.values
    # y_values = dfm.sds_change_intercept.values[0] + dfm.sds_change_rate.values[0] * x_values

    # logger.info("Regression line calculated")

    fig.add_trace(
        go.Scatter(
            # x=df["datetime"].iloc[0]
            # +(df["datetime"].iloc[-1] - df["datetime"].iloc[0])
            # * np.linspace(0, 1, 100),
            # y=y_pred,
            x=x_values,
            y=y_values,
            mode="lines",
            type="scatter",
            line=dict(color="grey"),
            name="Trend Line",
        )
    )

    logger.info("Scatter and regression line created")

    # Update layout
    fig.update_layout(
        title=dict(
            text="Transect timeseries",
            # automargin=True,
            y=0.98,
            font=dict(family="Segoe UI", size=24, weight="bold", color="#000000"),
            # yref="paper",
        ),
        plot_bgcolor="#ffffff",
        xaxis_title="Date [yr]",
        yaxis_title="Shoreline position [m]",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig.layout.xaxis.gridcolor = "#000000"
    fig.layout.yaxis.gridcolor = "#000000"
    fig.update_yaxes(
        zeroline=True,
        zerolinecolor="#ddd",
        zerolinewidth=0.5,
        showline=True,
        mirror=True,
        linecolor="#000000",
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#ddd",
    )
    fig.update_xaxes(
        zeroline=True,
        zerolinecolor="#ddd",
        zerolinewidth=0.5,
        showline=True,
        mirror=True,
        linecolor="#000000",
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#ddd",
    )

    fig.update_layout(
        legend=dict(
            orientation="h",  # horizontal legend
            yanchor="top",  # anchor legend to the top
            y=-0.13,  # position below the plot (negative = below)
            xanchor="center",  # center horizontally
            x=0.5,  # center alignment
        )
    )

    # define unique id based on current time
    id = str(int(time.time() * 1000))
    pltname = f"plot_{id}.html"

    # define htmlfile to write to serverside place and store
    htmlfile = os.path.join(_abspath, pltname)
    logger.info(f"Writing plot to: {htmlfile}")
    fig.write_html(htmlfile, auto_play=False)

    logger.info("Plot file created")

    assignmetadata(htmlfile, dfm)
    logger.info("Metadata added to plot")

    # based on the pltname, define the url needed to pass to frontend
    url = f"{_location}/data/{pltname}"
    logger.info(f"Generated URL: {url}")
    return url


# def assignmetadata(html, dfm):
#     """This function adds metadata to the scatterplot created in the previous routine.
#     This metadata gives overall description of the transect.

#     Args:


#          html (string): html file with scatterplot
#          dfm (dataframe): dataframe with basic metadata derived from gctr table

#     Returns:
#          html (string): overwrites the created html file by appending a specific portion of html code and contents
#     """
#     logger.info("Adding metadata to plot")

#     # Create a HTML string with information
#     html_info = """
#     <div>
#         <h2>Plot Information</h2>
#         <table>
#             <tr>
#                 <th>Transect id</th>
#                 <th>Country</th>
#                 <th>Continent</th>
#                 <th>Sandy</th>
#                 <th>Ambient change rate [m/yr]</th>
#             </tr>
#     """

#     for index, row in dfm.iterrows():
#         html_info += f"""
#             <tr>
#                 <td>{row["transect_id"]}</td>
#                 <td>{row["country"]}</td>
#                 <td>{row["continent"]}</td>
#                 <td>{str(row["sandy"])}</td>
#                 <td>{row["sds_change_rate"]}</td>
#             </tr>
#         """

#     html_info += """
#         </table>
#         </div>
#         """

#     # Read the plot HTML file
#     with open(html, "r", encoding="utf-8", errors="replace") as f:
#         plot_html = f.read()

#     # Add the information section to the plot HTML
#     with open(html, "w", encoding="utf-8") as f:
#         f.write(plot_html + html_info)

#     logger.info("Metadata successfully added")
#     return html


def assignmetadata(html, dfm):
    """Add a metadata table to the right-hand side of a Plotly HTML plot."""

    import logging

    logger = logging.getLogger(__name__)
    logger.info("Adding metadata to plot")

    # --- Build metadata table HTML (rows instead of columns) ---
    html_info = """
    <div class="info-container">
        <div class="header-with-icon">
            <h2>Transect metadata</h2>
            <a href="https://shorelinemonitor.earth/" target="_blank" class="info-button" title="More information about metadata">
                &#8505;
            </a>
        </div>
    """

    for _, row in dfm.iterrows():
        # classify change rate
        rate = row["sds_change_rate"]
        if rate < -5:
            change_class = "Extreme erosion (< -5 m/yr)"
        elif rate < -3.0 and rate >= -5.0:
            change_class = "Severe erosion (-3.0 to -5.0 m/yr)"
        elif rate < -1.0 and rate >= -3.0:
            change_class = "Intense erosion (-1.0 to -3.0 m/yr)"
        elif rate < -0.4 and rate >= -1.0:
            change_class = "Erosion (-0.4 to -1.0 m/yr)"
        elif rate >= -0.4 and rate <= 0.4:
            change_class = "Stable (-0.4 to 0.4 m/yr)"
        elif rate > 0.4 and rate <= 1.0:
            change_class = "Accretion (0.4 to 1.0 m/yr)"
        elif rate > 1.0 and rate <= 3.0:
            change_class = "Intense accretion (1.0 to 3.0 m/yr)"
        elif rate > 3.0 and rate <= 5.0:
            change_class = "Severe accretion (3.0 to 5.0 m/yr)"
        elif rate > 5:
            change_class = "Extreme accretion (> 5 m/yr)"

        html_info += """
        <table class="meta-table">
        """
        for label, value in [
            ("Transect ID", row["transect_id"]),
            ("Country", row["country"]),
            ("Continent", row["continent"]),
            # ("Coastal type", str(row["class_coastal_type"].replace("_", " "))),
            ("Shore type", str(row["class_shore_type"].replace("_", " "))),
            (
                "Ambient change rate [m/yr]",
                "%s +/- %s"
                % (
                    round(rate, 2),
                    round(row["sds_change_rate_std_err"], 2),
                ),
            ),
            ("Change rate classification", change_class),
            ("Linear fit R²", round(row["sds_r_squared"], 2)),
        ]:
            html_info += f"""
                <tr>
                    <th>{label}</th>
                    <td>{value}</td>
                </tr>
            """
        html_info += "</table>"

        # ⚠️ Add conditional warning if R² is small
        if (
            "sds_r_squared" in row
            and pd.notna(row["sds_r_squared"])
            and row["sds_r_squared"] < 0.5
        ):
            html_info += """
            <div class="warning-text">
                ⚠️ low conf. Linear Regression — use with caution ⚠️
            </div>
            """

        html_info += "<br>"

        # --- Feedback Section (Google Form) ---
    html_info += """
    <div class="feedback-container">

        <h3>User Feedback</h3>

        <label>
            <input type="radio" name="reliability" value="Reliable / representative">
            Reliable / representative
        </label><br>

        <label>
            <input type="radio" name="reliability" value="Questionable (interpret with caution)">
            Questionable (interpret with caution)
        </label><br>

        <label>
            <input type="radio" name="reliability" value="Not reliable">
            Not reliable
        </label>

        <div id="questionable-options" style="display:none; margin-top:10px;">

            <strong>Please specify:</strong><br>

            <label>
                <input type="checkbox" name="details"
                value="Residual outliers">
                Residual outliers
            </label><br>

            <label>
                <input type="checkbox" name="details"
                value="Time series discontinuities / step changes">
                Time series discontinuities / step changes
            </label><br>

            <label>
                <input type="checkbox" name="details"
                value="Dominance of short term variability (seasonal / event-scale)">
                Dominance of short term variability (seasonal / event-scale)
            </label><br>

            <label>
                <input type="checkbox" name="details"
                value="Coast is morphologically complex (e.g. tidal inlets, estuaries, deltas)">
                Coast is morphologically complex (e.g. tidal inlets, estuaries, deltas)
            </label><br>

            <label>
                <input type="checkbox" name="details"
                value="Human intervention likely (nourishment / structures / reclamation)">
                Human intervention likely (nourishment / structures / reclamation)
            </label><br>

            <label>
                <input type="checkbox" name="details"
                value="Imagery / processing artefacts likely (e.g. clouds, foam, turbid water)">
                Imagery / processing artefacts likely (e.g. clouds, foam, turbid water)
            </label><br>

            <label>
                <input type="checkbox" name="details"
                value="Datum / reference inconsistencies suspected">
                Datum / reference inconsistencies suspected
            </label><br>

            <label>
                <input type="checkbox" name="details" value="Other:">
                Other:
            </label>

            <div id="other-box" style="display:none; margin-top:5px;">
                <input type="text" id="other-text" placeholder="Please specify">
            </div>

        </div>

        <!-- Add hidden metadata fields here -->
        <input type="hidden" id="transect-id">
        <input type="hidden" id="center-lon">
        <input type="hidden" id="center-lat">
        <input type="hidden" id="country">
        <input type="hidden" id="continent">
        <input type="hidden" id="shore-type">
        <input type="hidden" id="change-rate">
        <input type="hidden" id="std">
        <input type="hidden" id="classification">
        <input type="hidden" id="r2">
        <input type="hidden" id="ip-address">

        <br>
        <button class="feedback-button" onclick="submitFeedback()">Submit</button>

    </div>
    """

    html_info += "</div>"

    # --- Read the Plotly HTML ---
    with open(html, "r", encoding="utf-8", errors="replace") as f:
        plot_html = f.read()

    # --- Strip HTML wrapper tags ---
    for tag in ["<html>", "</html>", "<head>", "</head>", "<body>", "</body>"]:
        plot_html = plot_html.replace(tag, "")

    # --- Construct new layout with auto-resize fix ---
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Plot with metadata</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #ffffff;
            }}
            .container {{
                display: flex;
                flex-direction: row;
                align-items: flex-start;
                gap: 0px;
                padding: 20px;
                width: calc(70% - 40px);
                height: calc(70vh - 40px);
                box-sizing: border-box;
            }}
            .plot-container {{
                flex: 3.5;
                min-width: 65%;
                overflow: hidden;
                align-self: flex-start;
                height: auto;
            }}
            .info-container {{
                flex: 2.0;
                max-width: 35%;
                background-color: #f8f9fa;
                border: 1px solid c;
                border-radius: 10px;
                padding: 15px 20px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                overflow-x: hidden;
                overflow-y: auto;
                max-height: 100%;
            }}
            h2 {{
                text-align: center;
                color: #333;
                margin-top: 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
                table-layout: auto; /* Let browser size columns naturally */
            }}
            th, td {{
                text-align: left;
                padding: 8px;
                border-bottom: 1px solid #ddd;
                vertical-align: top;
            }}
            th {{
                width: 1%;
                background-color: #f1f1f1;
                font-weight: 600;
                color: #333;
                white-space: nowrap; /* Prevent label wrapping */
            }}
            td {{
                white-space: normal; /* Values can wrap if long */
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            @media (max-width: 900px) {{
                .container {{
                    flex-direction: column;
                }}
                .info-container {{
                    max-width: 100%;
                }}
            }}
            .header-with-icon {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                margin-bottom: 2px;
            }}
            .header-with-icon h2 {{
                margin: 0;               /* remove default h2 margins */
                padding: 0;
                line-height: 1.5;        /* keep compact vertical spacing */
            }}
            .info-button {{
                text-decoration: none;
                font-size: 12px;
                color: #007bff;
                background-color: #e9f2ff;
                border-radius: 50%;
                width: 18px;
                height: 18px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                flex-shrink: 0;          /* keep the icon from shrinking */
            }}
            .info-button:hover {{
                background-color: #007bff;
                color: white;
            }}
            .warning-text {{
                color: #d9534f;              /* red tone */
                font-weight: 600;
                text-align: center;
                margin-top: 8px;
                font-size: 14px;
            }}
            .feedback-container {{
                margin-top: 15px;
                text-align: center;
            }}
            .feedback-button {{
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 14px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
            }}
            .feedback-button:hover {{
                background-color: #0056b3;
            }}
            .feedback-box textarea {{
                width: 100%;
                height: 70px;
                resize: vertical;
                padding: 6px;
                font-size: 13px;
            }}
            .feedback-box button {{
                margin-top: 5px;
                padding: 6px 10px;
                font-size: 13px;
            }}
            .feedback-box {{
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease;
            }}
            .feedback-box.active {{
                max-height: 200px;
            }}  
        </style>
    </head>
    <body>
        <div class="container">
            <div class="plot-container" id="plot-container">
                {plot_html}
            </div>
            {html_info}
        </div>

        <script>
            // Wait for page to load, then trigger Plotly resize
            window.addEventListener("load", function() {{
                if (window.Plotly) {{
                    const plots = document.querySelectorAll('.js-plotly-plot');
                    plots.forEach(plot => {{
                        Plotly.Plots.resize(plot);
                    }});
                }}
            }});

            // --- Show conditional options ---
            document.addEventListener("change", function(e) {{

                if (e.target.name === "reliability") {{
                    const show = e.target.value === "Questionable (interpret with caution)";
                    document.getElementById("questionable-options").style.display = show ? "block" : "none";
                }}

                if (e.target.name === "details" && e.target.value === "Other:") {{
                    document.getElementById("other-box").style.display =
                        e.target.checked ? "block" : "none";
                }}
            }});

            // Fill hidden metadata field for potential use in form submission
            document.getElementById("transect-id").value = "{dfm['transect_id'].values[0]}";
            document.getElementById("center-lon").value = "{dfm['center_lon'].values[0]}";
            document.getElementById("center-lat").value = "{dfm['center_lat'].values[0]}";
            document.getElementById("country").value = "{dfm['country'].values[0]}";
            document.getElementById("continent").value = "{dfm['continent'].values[0]}";
            document.getElementById("shore-type").value = "{dfm['class_shore_type'].values[0]}";
            document.getElementById("change-rate").value = "{dfm['sds_change_rate'].values[0]}";
            document.getElementById("std").value = "{dfm['sds_change_rate_std_err'].values[0]}";
            document.getElementById("classification").value = "{change_class}";
            document.getElementById("r2").value = "{dfm['sds_r_squared'].values[0]}";

            // Optional: get IP
            fetch("https://api.ipify.org?format=json")
                .then(response => response.json())
                .then(data => {{
                    document.getElementById("ip-address").value = data.ip;
                }});

            // --- Submit to Google Form ---
            function submitFeedback() {{

                const selected = document.querySelector('input[name="reliability"]:checked');

                if (!selected) {{
                    alert("Please select an assessment option.");
                    return;
                }}

                const formUrl = "https://docs.google.com/forms/d/e/1FAIpQLSdYunKIvdp2EjryKVTnXTFsTINsI_DUKQX7FmVpC6e7sHbx2Q/formResponse";
                const formData = new FormData();

                // Question 1
                formData.append("entry.1968241590", selected.value);

                // If Questionable → require at least one checkbox
                if (selected.value.includes("Questionable")) {{

                    const checkboxes = document.querySelectorAll('input[name="details"]:checked');

                    if (checkboxes.length === 0) {{
                        alert("Please select at least one reason.");
                        return;
                    }}

                    let isValid = true;   // <-- control flag

                    checkboxes.forEach(cb => {{

                        if (cb.value === "Other:") {{

                            const otherText = document.getElementById("other-text").value.trim();

                            if (otherText === "") {{
                                alert("Please specify the 'Other' reason.");
                                isValid = false;
                                return;
                            }}

                            formData.append("entry.467815806", "__other_option__");
                            formData.append("entry.467815806.other_option_response", otherText);

                        }} else {{
                            formData.append("entry.467815806", cb.value);
                        }}
                    }});

                    // 🚨 Stop submission completely if invalid
                    if (!isValid) {{
                        return;
                    }}
                }}

                // append additional hidden metadata fields if needed
                formData.append("entry.2142836349", document.getElementById("transect-id").value); // Hidden transect ID
                formData.append("entry.51855834", document.getElementById("center-lon").value); // Hidden center longitude
                formData.append("entry.344849078", document.getElementById("center-lat").value); // Hidden center latitude
                formData.append("entry.659486804", document.getElementById("country").value); // Hidden country
                formData.append("entry.1029413259", document.getElementById("continent").value); // Hidden continent
                formData.append("entry.346152434", document.getElementById("shore-type").value); // Hidden shore type
                formData.append("entry.934577245", document.getElementById("change-rate").value); // Hidden change rate
                formData.append("entry.2092884113", document.getElementById("std").value); // Hidden standard deviation
                formData.append("entry.357345898", document.getElementById("classification").value); // Hidden classification
                formData.append("entry.1030618903", document.getElementById("r2").value); // Hidden R-squared
                formData.append("entry.290909900", document.getElementById("ip-address").value); // Hidden IP address

                fetch(formUrl, {{
                    method: "POST",
                    mode: "no-cors",
                    body: formData
                }});

                alert("Thank you for your submission!");
            }}

        </script>
    </body>
    </html>
    """

    # --- Save final HTML ---
    with open(html, "w", encoding="utf-8") as f:
        f.write(full_html)

    logger.info(
        "✅ Metadata successfully added on the right-hand side (auto-resizing fixed)."
    )
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
                ST_X(ST_Centroid(geometry)) as center_lon,
                ST_Y(ST_Centroid(geometry)) as center_lat,
                sds_change_rate,
                sds_change_intercept,
                sds_change_rate_std_err,
                sds_r_squared,
                class_shore_type,
                class_coastal_type,
                index
                FROM public.gctr g
                join country c on c.idccn = g.idccn
                join continent ct on ct.idcnt = g.idcnt
                where index = {profile}"""

    df = pd.read_sql_query(strsql, _engine)
    logger.info(f"Metadata query returned {len(df)} rows")

    logger.info("Executing profile data query")
    strsql = f"""select datetime, shoreline_position, obs_is_primary, obs_is_outlier 
                 from shorelinemonitor_series 
                 where gctr_id = '{profile}'
                 order by datetime"""

    dfp = pd.read_sql_query(strsql, _engine)
    logger.info(f"Profile data query returned {len(dfp)} rows")

    # Handle case where no data is available
    if len(dfp) == 0:
        logger.warning(f"No derived measurements available for profile: {profile}")
        return None  # or return an error message

    logger.info("Generating scatterplot")
    url = scatterplot(dfp, df)
    logger.info(f"=== Handler completed successfully, returning URL: {url} ===")
    return url


def test():
    """Test function"""
    print(handler(2422064))


if __name__ == "__main__":
    test()
