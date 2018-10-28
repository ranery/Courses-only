# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import sys
import data
from Apriori import Apriori
from optparse import OptionParser

optparser = OptionParser()
optparser.add_option('--inputFile', dest='input', help='filename containing csv', default='goods.csv')
optparser.add_option('--minSupport', dest='minS', help='minimum support value', default=0.5, type='float')
optparser.add_option('--minConfidence', dest='minC', help='minimum confidence value', default=0.5, type='float')
(options, args) = optparser.parse_args()

# get dataset
inputFile = None
if options.input is not None:
    inputFile = data.dataset(options.input)
else:
    sys.exit('No dataset filename specified, system exit!')

# apriori
Apriori = Apriori()
items, rules = Apriori.run(inputFile, options.minS, options.minC)
Apriori.printResults(items, rules)