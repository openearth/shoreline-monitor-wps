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

import flask

import pywps
from pywps.app.Service import Service
import logging

# from flask_cors import CORS

# Ultimate question
from processes.wps_ultimate_question import UltimateQuestion
from processes.wps_shoreline_getprofile import WpsShorelineGetprofile
from processes.wps_shoreline_getstats import WpsShorelineGetStats


# TODO add the proces in the processes list
processes = [
    UltimateQuestion(),
    WpsShorelineGetprofile(),
    WpsShorelineGetStats(),
]

# Description used in template
process_descriptor = {}
for process in processes:
    abstract = process.abstract
    identifier = process.identifier
    process_descriptor[identifier] = abstract

service = Service(processes, ["pywps.cfg"])
application = flask.Flask(__name__)
# CORS(application)


@application.route("/")
def hello():
    server_url = pywps.configuration.get_config_value("server", "url")
    request_url = flask.request.url
    return flask.render_template(
        "index.html",
        request_url=request_url,
        server_url=server_url,
        title="PyWPS Web Processing Service",
        process_descriptor=process_descriptor,
    )


@application.route("/wps", methods=["GET", "POST"])
def wps():
    return service


@application.route("/data/" + "<path:filename>")
def outputfile(filename):
    targetfile = os.path.join("data", filename)
    if os.path.isfile(targetfile):
        file_ext = os.path.splitext(targetfile)[1]
        with open(targetfile, mode="rb") as f:
            file_bytes = f.read()
        mime_type = None
        if "xml" in file_ext:
            mime_type = "text/xml"
        return flask.Response(file_bytes, content_type=mime_type)
    else:
        flask.abort(404)


@application.route("/static/" + "<path:filename>")
def staticfile(filename):
    targetfile = os.path.join("static", filename)
    if os.path.isfile(targetfile):
        with open(targetfile, mode="rb") as f:
            file_bytes = f.read()
        mime_type = None
        return flask.Response(file_bytes, content_type=mime_type)
    else:
        flask.abort(404)


if __name__ == "__main__":
    import argparse

    # application.run(debug=True)
    parser = argparse.ArgumentParser(
        description="""Script for starting an example PyWPS
                       instance with sample processes""",
        epilog="""Do not use this service in a production environment.
         It's intended to be running in test environment only!
        For more documentation, visit http://pywps.org/doc
        """,
    )
    parser.add_argument(
        "-d", "--daemon", action="store_true", help="run in daemon mode"
    )
    parser.add_argument(
        "-a",
        "--all-addresses",
        action="store_true",
        help="run flask using IPv4 0.0.0.0 (all network interfaces),"
        + "otherwise bind to 127.0.0.1 (localhost).  This maybe necessary in systems that only run Flask",
    )
    args = parser.parse_args()

    if args.all_addresses:
        bind_host = "0.0.0.0"
    else:
        bind_host = "127.0.0.1"

    if args.daemon:
        pid = None
        try:
            pid = os.fork()
        except OSError as e:
            raise Exception("%s [%d]" % (e.strerror, e.errno))

        if pid == 0:
            os.setsid()
            application.run(threaded=True, host=bind_host)
        else:
            os._exit(0)
    else:
        application.run(threaded=True, host=bind_host)
