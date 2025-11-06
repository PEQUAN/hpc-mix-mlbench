# coding=utf-8
# This file is part of PROMISE.
#
# 	PROMISE is free software: you can redistribute it and/or modify it
# 	under the terms of the GNU Lesser General Public License as
# 	published by the Free Software Foundation, either version 3 of the
# 	License, or (at your option) any later version.
#
# 	PROMISE is distributed in the hope that it will be useful, but WITHOUT
# 	ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# 	or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
# 	Public License for more details.
#
# 	You should have received a copy of the GNU Lesser General Public
# 	License along with PROMISE. If not, see
# 	<http://www.gnu.org/licenses/>.
#
# Promise v1 was written by Romain Picot
# Promise v2 has been written from v1 by Thibault Hilaire and Sara Hoseininasab
# Promise v3 has been written from v2 by  Thibault Hilaire, Fabienne JÉZÉQUEL and Xinye Chen
#   Promise v3 enables version check, pip installing, arbitrary precision, etc., features.
# 	  Sorbonne Universitéx, LIP6 (Computing Science Laboratory), Paris, France. 
#     Contact: thibault.hilaire@lip6.fr, fabienne.jezequel@lip6.fr, xinyechenai@gmail.com
#
# 	contain the entry function, called to run Promise
#
# 	© Thibault Hilaire and Fabienne JÉZÉQUEL, April 2024


"""
activate-promise

Usage:
    activate-promise [[--CC=<compiler1>|--CXX=<compiler2>] | [--CC=<compiler1> --CXX=<compiler2>]]
    activate-promise (-h | --help)
    activate-promise (--version)
    deactivate-promise 
    deactivate-promise (-h | --help)
    deactivate-promise (--version)

Options:
	-h --help                     Show this screen.
    --version                     Show version.
    --CC=<compiler1>        	  Set compiler for C program [default: g++].
    --CXX=<compiler2>             Set compiler for C++ program [default: g++].
"""


import os
import sys
# from .utils import customArgParser, __params__
from docopt import docopt
from .utils import get_version

cachePath = "/cache"

# ./configure CXX=YourC++Compiler 
# ./configure --prefix=TheInstallationDirectory
# ./configure CXX=YourC++Compiler  --prefix=TheInstallationDirectory 
# ./configure --prefix=`pwd` --enable-fortran

# Macos 
# Brew gcc/7.3.0_1 (gcc, g++, gfortran)
# ./configure --prefix=`pwd` CC=gcc-7 CXX=g++-7 --enable-fortran
# ./configure --prefix=`pwd` CC=clang CXX=clang++ OPENMP_CFLAGS="-I/usr/local/Cellar/libomp/5.0.1/include -Xclang -fopenmp" OPENMP_CXXFLAGS="-I/usr/local/Cellar/libomp/5.0.1/include -Xclang -fopenmp" OPENMP_LDFLAGS="-L/usr/local/Cellar/libomp/5.0.1/lib -lomp"
# ./configure --prefix=`pwd` CC=clang CXX=clang++ OPENMP_CFLAGS="-I/usr/local/Cellar/libomp/7.0.0/include -Xclang -fopenmp" OPENMP_CXXFLAGS="-I/usr/local/Cellar/libomp/7.0.0/include -Xclang -fopenmp" OPENMP_LDFLAGS="-L/usr/local/Cellar/libomp/7.0.0/lib -lomp"
# On OSX gcc may be a wrapper for clang. In that case you can use:
# ./configure --prefix=`pwd` CC=clang CXX=clang++

def activate(argv=None): 
    
    import subprocess
    import platform

    opt_system = platform.system()
    curr_loc = os.path.dirname(os.path.realpath(__file__))

    install_cadna = True
    # args = customArgParser(sys.argv[1:] if argv == {} else argv, __params__)      # parse the command line

    args = docopt(__doc__, argv=sys.argv[1:] if argv is {} else argv, 
                  version=get_version(os.path.dirname(os.path.realpath(__file__))+'/__init__.py')
                  )

    compiler_CXX = args['--CXX']
    compiler_CC = args['--CC']
        
    if not os.path.exists(curr_loc + cachePath):
        os.makedirs(cachePath)
        
    if 'CADNA_PATH' in os.environ:  # check environmental variables
        import logging
        logging.basicConfig()
        log = logging.getLogger()

        log.warning("It looks like your machine has CADNA installed, are you sure to proceed CADNA installation? ")
        check_point = input("Please answer 'yes' or 'no':")

        if check_point.lower() in {'yes', 'y'}:
            install_cadna = True

        else:
            install_cadna = False

    curr_loc = os.path.dirname(os.path.realpath(__file__))

    if os.path.exists(curr_loc + cachePath):
        if os.path.isfile(curr_loc + cachePath + '/CXX.txt'):
            with open(curr_loc+cachePath+"/CXX.txt", "r") as file:
                compiler = file.read().replace('\n', '')
                print('check compilers:', compiler)

                if compiler_CXX != compiler:
                    install_cadna = True
                    

    os.chdir(curr_loc+'/cadna')

    if install_cadna: # install cadna if a.out and libcadnaC.a is not found
        with open(curr_loc + cachePath + "/CC.txt", "w") as file:
            file.write(compiler_CC)
            
        with open(curr_loc + cachePath + "/CXX.txt", "w") as file:
            file.write(compiler_CXX)

        # must set environmental variables
        if not os.path.isfile('a.out') and not os.path.isfile('lib/libcadnaC.a'):
            
            if opt_system in {'Linux', 'posix', 'Darwin'}:
                if compiler_CC == 'g++' and compiler_CXX == 'g++':
                    subprocess.call('bash run_unix.sh', shell=True)

                elif compiler_CC != 'g++' and compiler_CXX != 'g++':
                    subprocess.call('bash run_unix.sh '+compiler_CC+' '+compiler_CXX, shell=True)

                elif compiler_CC != 'g++' and compiler_CXX == 'g++':
                    subprocess.call('bash run_unix_cc.sh '+compiler_CC, shell=True)

                else:
                    subprocess.call('bash run_unix_cxx.sh '+compiler_CXX, shell=True)

    


def deactivate(): 
    import subprocess
    curr_loc = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(curr_loc + cachePath):
        os.makedirs(cachePath)

    curr_loc = os.path.dirname(os.path.realpath(__file__))
    os.chdir(curr_loc+'/cadna')
    subprocess.call('make clean', shell=True)
    subprocess.call('bash clean.sh', shell=True)