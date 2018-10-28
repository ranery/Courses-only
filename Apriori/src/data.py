# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""

def dataset(fname):
    f = open(fname, 'r')
    for line in f:
        record = line.strip().split(',')
        yield record