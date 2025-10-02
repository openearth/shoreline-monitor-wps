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

# http://localhost:5000/wps?service=wps&request=Execute&version=2.0.0&Identifier=wps_shoreline_getprofile&datainputs=profileid={"profileid":2805066}
# 
# production:
# getcapabilities https://shoreline-monitor.avi.directory.intra/wps?service=wps&request=getcapabilities
# execute https://shoreline-monitor.avi.directory.intra/wps?service=wps&request=Execute&version=2.0.0&Identifier=wps_shoreline_getprofile&datainputs=profileid={"profileid":2805066}


import json
from pywps import Format
from pywps.app import Process
from pywps.inout.inputs import ComplexInput
from pywps.inout.outputs import ComplexOutput
from pywps.app.Common import Metadata
from .shoreline_getprofile import handler
import logging
logger = logging.getLogger("PYWPS")

class WpsShorelineGetprofile(Process):
    def __init__(self):
        inputs = [ComplexInput('profileid', 'Profile Identification as profileid',
                         supported_formats=[Format('application/json')])
        ]
        outputs = [
            ComplexOutput(identifier="profileinformation", title="Profile information", supported_formats=[Format('application/json')])
        ]

        super(WpsShorelineGetprofile, self).__init__(
            self._handler,
            identifier="wps_shoreline_getprofile",
            version="1.0.0",
            title="Retrieve shoreline information",
            abstract='The process provides information about the change rate of shorelines. The source' \
            ' of the data is a long time series of satellite imagery (from 1984 till now) that is analysed on Google Earth Engine' \
            ' and provides information on accreation and decretion rates on yearly basis',
            profile="",
            metadata=[
                Metadata("WpsShorelineGetprofile"),
                Metadata("Accreation, decretion data as timeseries"),
            ],
            inputs=inputs,
            outputs=outputs,
            store_supported=False,
            status_supported=False,
        )

    def _handler(self, request, response):
        """Returns a complex output (json with link to the document with a timeseries plot 
           of accretion/decreation number of coastlines) from shorelinemonitor_series table

        Args:
            request (json):  request is composed of profile-id that is passed from front end to PyWPS, 
                             profile-id is the index of gctr table
            response (json): json with a link to a html document with plot of shoreline monitor timeseries

        Returns:
            json: json with a link to a html document with plot of shoreline monitor timeseries
        """
        try:
            profileid = request.inputs["profileid"][0].data
            profileid_json = json.loads(profileid)
            logger.info(f'provided input: {profileid_json["profileid"]}')
            url = handler(profileid_json['profileid'])
            logger.info(f'url created: {url}')
            response.outputs['profileinformation'].data = json.dumps({'url':url})
            
        except Exception as e:
            res = { 'errMsg' : 'ERROR: {}'.format(e)}
            print(res)
            response.outputs["profileinformation"].data = "Something went very wrong, please check logfile"
        return response
