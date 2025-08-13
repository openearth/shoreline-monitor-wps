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

from pywps.app import Process
from pywps.inout.outputs import LiteralOutput
from pywps.app.Common import Metadata


class UltimateQuestion(Process):
    def __init__(self):
        inputs = []
        outputs = [
            LiteralOutput("answer", "Answer to Ultimate Question", data_type="string")
        ]

        super(UltimateQuestion, self).__init__(
            self._handler,
            identifier="ultimate_question",
            version="1.0.0",
            title="Answer to the ultimate question",
            abstract='The process gives the answer to the ultimate question\
             of "What is the meaning of life?',
            profile="",
            metadata=[
                Metadata("Ultimate Question"),
                Metadata("What is the meaning of life"),
            ],
            inputs=inputs,
            outputs=outputs,
            store_supported=False,
            status_supported=False,
        )

    def _handler(self, request, response):
        response.outputs["answer"].data = "42"
        return response
