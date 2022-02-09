#!/usr/bin/env python
# -*- coding: utf-8 -*-

from  pynmd.models.adcirc.pre import adcirc_pre as adcp
from  base_info import datum_fname

adcp.fort14_to_nc(datum_fname)
