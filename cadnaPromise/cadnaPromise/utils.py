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
# Promise v3 has been written from v2 by Thibault Hilaire, Fabienne JÉZÉQUEL and Xinye Chen
#   Promise v3 enables version check, pip installing, arbitrary precision, etc., features.
# 	  Sorbonne Universitéx, LIP6 (Computing Science Laboratory), Paris, France. 
#     Contact: thibault.hilaire@lip6.fr, fabienne.jezequel@lip6.fr, xinyechenai@gmail.com
#
# 	contain the entry function, called to run Promise
#
# 	© Thibault Hilaire and Fabienne JÉZÉQUEL, April 2024


import json
import copy
from os import chdir
from os.path import exists, join
from yaml import load as loadyml, SafeLoader, YAMLError
from glob import glob
from subprocess import list2cmdline, STDOUT, check_output, CalledProcessError
from timeit import default_timer
from tqdm import tqdm
import os
from colorama import Fore

import shlex

from .errors import PromiseError
from .logger import PrLogger, getInt


logger = PrLogger()        # logger



def str2dict(s):
	"""transform a string in form 'param=value' to a dictionary {'param':'value'} """
	# see https://stackoverflow.com/questions/23228352/turn-key-value-string-into-a-dict
	return dict(x.split('=') for x in shlex.split(s))

def commaAnd(li):
	"""Returns a string where all the items of the list li are separated with a comma `,`
	except for the last one separated by `and` (and of cours there is no `and` if the
	list has only one item"""
	return ", ".join(li[0:-1]) + (' and ' if li[0:-1] else '') + li[-1]


def runCommand(cmd, alias={}, errorsAsDegug=False):
	"""run shell command and log it
	(and manage the errors)
	Returns True if everything was ok
	log the errors as debug message when errorsAsDebug is True"""
	# proceed to replacement in the command line according to the dictionary alias
	cmdline = list2cmdline(cmd)
	for k, v in alias.items():
		cmdline = cmdline.replace(k, v)
	# run the command
	try:
		logger.command('$ ' + cmdline)
		lines = check_output(cmdline, stderr=STDOUT, shell=True, universal_newlines=True,)
	except CalledProcessError as e:
		if errorsAsDegug:
			logger.command(e.output)
		else:
			logger.error(e.output)
		return False, ''
	# log the lines
	if lines:
		# display the line in `outputPromise` if it begins with `[PROMISE_`,  otherwise in `output`
		for li in lines.split('\n'):
			if li.startswith("[PROMISE_"):
				logger.outputPromise(li)
			else:
				logger.output(li)

	return True, lines.split('\n')


def cd(path):
	"""Change directory"""
	logger.command('$ cd '+path)
	chdir(path)




def getYMLOptions(args):
	"""get the options from the yaml file
	and merge them with those from the command line
	Returns a dictionary of the options {name:value}
	"""
	# read the config file (`promise.ymf` file  or the file given with '--conf' arg)
	ymlFileName = args['--conf'] if args['--conf'] else 'promise.yml'
	try:
		with open(ymlFileName, 'r') as f:
			dataset = loadyml(f.read(), SafeLoader)

	except IOError as err:
		if args['--conf']:
			logger.error("The file %s is empty or doesn't exist", ymlFileName)
			raise PromiseError(err) from err
		
	except YAMLError as err:
		logger.error("Cannot parse the %s file", ymlFileName)
		raise PromiseError(err) from err
	#
	conf = {'--'+key: True if value is None else value for key, value in dataset.items()}
	# add a file handler to the logger if it's in conf but not in args
	if '--log' in conf and not args['--log']:
		# get the path
		YMLpath = conf['--path'] if '--path' in conf else ''
		path = args['--path'] if args['--path'] else YMLpath
		# get the verbosity for the lg
		if not args['--verbosityLog']:
			verbosityLog = getInt(logger, conf, 'verbosityLog', 1, 4, default=int(args['--verbosity']))
		else:
			verbosityLog = args['--verbosityLog']
		# add the file logger
		logger.addFileLogger(path, conf['--log'], verbosityLog)

	# merge the two
	return {str(key): args.get(key) or conf.get(key) for key in set(conf) | set(args)}


def getFPM(args):
	"""get the floating point number format from fp.json
	"""
	fpfmt_reference	= { 'b': [8, 7],
						'h': [5, 10],
						's': [8, 23],
						'd': [11, 32],
						'q': [15, 112],
						'o': [19, 236]
						}
	
	fileName = args['--fp'] if args['--fp'] is not None else 'fp.json'

	try:
		with open(fileName, 'r') as file:
			fpfmt = json.load(file)

		for i in fpfmt:
			if i not in fpfmt_reference:
				fpfmt_reference[i] = fpfmt[i]

		fpfmt_reference = {k: v for k, v in sorted(
			fpfmt_reference.items(), key=lambda item: item[1][1])}
	
	except FileNotFoundError:
		#import logging
		#logging.basicConfig()
		#log = logging.getLogger(__file__)

		# logger.message("File for customized precision formats is lacking.")
		pass

	return fpfmt_reference



def sort_precs(method, fpfmt):
	"""Sort floating points in terms of its unit-roundoff."""
	precs = ''
	for p in fpfmt:
		if p in method:
			precs = precs + p

	return precs


def update_types(types, typeNames, fpfmt):
	"Update floating point type definitions and names."
	types_copy = copy.deepcopy(types)
	typeNames_copy = copy.deepcopy(typeNames)
	for p in fpfmt:
		if p not in types:
			types_copy[p] = 'flx::floatx<'+str(fpfmt[p][0]) +', '+ str(fpfmt[p][1]) +'>'
			typeNames_copy[p] = 'custom('+str(fpfmt[p][0]) +', '+ str(fpfmt[p][1])+')'

	return types_copy, typeNames_copy



def parseOptions(options):
	"""parse the dictionary of options
	Returns:
		- method: (string) method name for the mixed-precision ('hs', 'sd' or 'hsd')
		- path: (string) the path of the project
		- files: (list of string) list of filenames
		- run: (string) file to run
		- nbDigits: (int) nb of digits
		- compileLines: (list of string) commands to compile the project
		- output: (string) folder where put the generated code
		- typeCustom: (dictionary) custom types (used with __PRC_xxxx__)
		- alias: (string)
		"""
	# get the path
	path = options['--path'] if options['--path'] else ''

	# get the output path
	output = options['--output'] if options['--output'] else 'result'

	# list of files to examin (given by --files, or the *.c files in the path)
	if options['--files']:
		files = [name.strip() for name in options['--files'].split(',')]
	else:
		files = glob('*.c')
	for f in files:
		if not exists(join(path, f)):
			logger.error("The file %s doesn't exist in %s (declared in `--files` option)", f, path)
			raise PromiseError

	# run file
	run = options['--run']
	if not run:
		logger.error('The executable name is missing (`--run` option)')
		raise PromiseError

	# nb of digits
	try:
		nbDigitsGen = int(options['--nbDigits'])
	except ValueError as err:
		logger.error('The number of digits is missing or incorrect (`--nbDigits` option)')
		raise PromiseError(err) from err
	if '--nbDigitsPerVariable' in options:
		try:
			nbDigitsPerVar = {var: int(d) for var, d in options['--nbDigitsPerVariable'].items()}
		except (ValueError, AttributeError) as err:
			logger.error('The option `nbDigitsPerVariable` is incorrect. It should be a yml dictionary {varName: nbDigits}')
			raise PromiseError() from err
	else:
		nbDigitsPerVar = {}

	# compile commands
	if options['--compile']:
		compileLines = options['--compile']
	else:
		logger.error("No compilation command is given (`--compile` option)")
		raise PromiseError



	# method
	method = options['--precs'] # [i for i in options if '--' not in i and (i != '--version' or i != '--v')][0].lower()
	
	# alias
	if options['--alias'] is not None:
		if '=' in options['--alias']: # retirn dict
			alias = str2dict(options['--alias']) if options['--alias'] else {}
		else:
			alias = options['--alias'] if options['--alias'] is not None else {} # retirn str
	else:
		alias = {}
	
	# relative error threshold
	relErrorThres = -1 if not options['--relError'] else float(options['--relError'])


	return method, path, files, run, (nbDigitsGen, nbDigitsPerVar), relErrorThres, compileLines, output, options.get('--custom', None), alias
# method, path, files, run, (nbDigitsGen, nbDigitsPerVar), compileLines, output, options.get('--custom', None), alias




def pause(status=None):
	"""Make a pause
	wait for the user to press a key"""
	if status is None:
		tqdm.write("Press any key to continue...")
	else:
		status.set_description_str("%sPress any key to continue..." % Fore.RESET)
		status.refresh()
	os.system("""bash -c 'read -s -n 1'""")



def get_version(fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("__version__ = '"):
                return line.split("'")[1]
    raise RuntimeError('Error in parsing version string.')






class Timing:
	"""Small context manager to measure the time used
	it also increased the list [nbOccurenceTotal, nbOccurenceFailed, time] that is used for the constructor
	To be used as:

	>>> executions = [0,0]
	>>> with Timing(executions):
	>>>     ...
	>>>     #code_we_want_to_measure_time()
	>>>     ...

	and the elements of the list executions will be taken update accordingly

	We use here timeit.default_timer for the time (and not perf_counter or process_time)
	See https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python/25823885 for example"""
	def __init__(self, nbTi=None):
		"""nbTi is a 2-element list [nbOccurenceTotal, nbOccurenceFailed, time]
		those elements will be increased accordingly"""
		self._nbTi = nbTi
		self._timer = 0         # we haven't start yet
		self._timing = 0       # we haven't measure anything

	@property
	def timing(self):
		"""Return the time passed in the context"""
		return self._timing

	def __enter__(self):
		"""Enter in the context"""
		self._timer = default_timer()
		return self     # used to get the timing, after the context

	def __exit__(self, ty, value, traceback):
		""""We quit the context"""
		self._timing = default_timer() - self._timer
		if self._nbTi:
			self._nbTi[2] += self._timing
			self._nbTi[0] += 1
			if ty:
				self._nbTi[1] += 1




### the following is used as test and simulations
__params__ =  {'--alias': None,
             '--compile': None,
             '--conf': 'promise.yml',
             '--debug': False,
             '--files': None,
             '--help': False,
             '--log': None,
             '--nbDigits': None,
             '--parsing': True,
             '--output': None,
             '--path': None,
             '--pause': False,
             '--run': None,
             '--verbosity': '1',
             '--verbosityLog': None,
			 '--CC': 'g++',
			 '--CXX': 'g++',
			 '--fp' : None,
			 '--version': None,
			 '--v': None
}


def customArgParser(args, params):
	"Return a user-defined argment list for PROMISE settings."
	params_copy = copy.deepcopy(params)
	args_clean = [i.split('=') for i in args]

	for i in args:
		if '--alias' in i:
			args_clean.append(['--alias', i.replace('--alias=', '')])
		else:
			args_clean.append(i.split('='))
		
	for i in args_clean:
		if '--' in i[0] and i[0] != '--version' and i[0] != '--v':
			if i[1] in {'True', 'true'}:
				params_copy[i[0]] = True
			elif i[1] in {'False', 'false'}:
				params_copy[i[0]] = False
			else:
				params_copy[i[0]] = i[1]
				
		elif i[0] == '--version' or i[0] == '--v':
			pass

		else:
			params_copy[i[0]] = True

	return params_copy
