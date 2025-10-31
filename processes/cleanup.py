# -*- coding: utf-8 -*-
# Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2025 Deltares
#       Gerrit Hendriksen
#       gerrit.hendriksen@deltares.nl
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

#script to clean up tmp files of PyWPS processes
import os
import shutil
from shoreline_getprofile import _initialize_config
import sys

abspath =  _initialize_config()

def cleanup_pywps_tmp(tmp_dir):
    if not os.path.exists(tmp_dir):
        print(f"PyWPS tmp directory not found: {tmp_dir}")
        return

    print(f"\n▶ Cleaning up PyWPS temporary files in: {tmp_dir}")
    for entry in os.listdir(tmp_dir):
        path = os.path.join(tmp_dir, entry)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                print(f"  Deleted file: {entry}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  Deleted directory: {entry}")
        except Exception as e:
            print(f"  Failed to delete {entry}: {e}")

if __name__ == "__main__":
    try:
        tmp_dir = abspath
        cleanup_pywps_tmp(tmp_dir)
    except Exception as e:
        print(f"Error during PyWPS temporary files cleanup: {e}")
