# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import numpy as np

def weight_init(f, c, hh, ww):
    w = 0.01 * np.random.randn(f, c, hh, ww)
    b = 0.01 * np.random.randn(f, 1)
    return w, b