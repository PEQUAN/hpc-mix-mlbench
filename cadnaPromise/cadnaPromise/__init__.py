from .promise import Promise 
from .run import runPromise

import os
__version__ = '0.3.3'

curr_loc = os.path.dirname(os.path.realpath(__file__))


cachePath = "/cache"
__compiler__ = 'g++'

if os.path.exists(curr_loc + cachePath):
    if os.path.isfile(curr_loc + cachePath + '/.CXX.txt'):
        with open(curr_loc+cachePath+"/CXX.txt", "r") as file:
            __compiler__ = file.read()

