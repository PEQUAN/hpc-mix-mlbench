# coding=utf-8

"""This file is part of PROMISE.

	PROMISE is free software: you can redistribute it and/or modify it
	under the terms of the GNU Lesser General Public License as
	published by the Free Software Foundation, either version 3 of the
	License, or (at your option) any later version.

	PROMISE is distributed in the hope that it will be useful, but WITHOUT
	ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
	or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
	Public License for more details.

	You should have received a copy of the GNU Lesser General Public
	License along with PROMISE. If not, see
	<http://www.gnu.org/licenses/>.

Promise v1 was written by Romain Picot
Promise v2 has been written from v1 by Thibault Hilaire and Sara Hoseininasab
Promise v3 has been written from v2 by Thibault Hilaire, Fabienne JÉZÉQUEL and Xinye Chen
	Promise v3 enables version check, pip installing, arbitrary precision, etc., features.
	Sorbonne Université
	LIP6 (Computing Science Laboratory)
	Paris, France. Contact: thibault.hilaire@lip6.fr, fabienne.jezequel@lip6.fr, xinyechenai@gmail.com

© Thibault Hilaire and Fabienne JÉZÉQUEL, April 2024
"""

from tempfile import mkdtemp
from os.path import join
from os import getcwd, environ#, chmod
from stat import S_IXUSR
from pkg_resources import resource_filename
import re
from tqdm import tqdm
from math import ceil, log2, isnan, isinf
from colorama import Fore
from datetime import datetime

from .errors import PromiseError, PromiseCompilationError
from .utils import runCommand, cd, Timing, pause, commaAnd
from .prfile import PrFile


from .logger import PrLogger
from os.path import join, split

logger = PrLogger()


regDumpST = re.compile(
	r"^\[PROMISE_DUMP_ST\] (\w+)(?:\[(\d+)\])? = \(([\w\-+.]+),([\w\-+.]+),([\w\-+.]+)\), nb significant digits=(\d+)")
regDump = re.compile(r"^\[PROMISE_DUMP\] (\w+)(?:\[(\d+)\])? = ([\w\-+.]+)")



class Promise:
	"""
	principal class for Promise v2
	It mainly contains a list of files (list of PrFile objects) and a dictionary, that maps for every variable type
	considered by Promise (all the types in form __PROMISE__ or __PR_xxx__ in the C code) to a given type
	"""

	def __init__(self, path, files, run, nbDigits, compileLines, parsing=False, alias={}):
		"""Build a Promise object
		Parameters:
			- path: (strin) path where are the files
			- files: (list) filenames
			- run: (string) filename of the executable
			- nbDigits: (tuple (int,{string:int})) number of correct digits required and then nb of digits per variable
			- compileLines: (list of string) command lines to execute to create the `run` file
			- parsing: (boolean) True if we cleverly parse the C files
		"""
		self._path = path
		self._run = run
		self._nbDigits = nbDigits
		self._compile = compileLines
		self._types = {}            # key: 0, 1, 2 for __PROMISE__ defined types and 'xxx' for __PR_xxx__ types
		self._counter = 0           # index of the *next* __PROMISE__ type
		self._ref = {}              # dictionary of references (variable name: mean value or list of mean values)
		self._power = {}            # dictionary of 10^-nbDigits
		self._variables = {}        # dictionary of variables {typeName: (variableName, fileName, lineNumber)}
		self._compilations = [0, 0, 0]
		self._executions = [0, 0, 0]
		self._cache = {}
		self._alias = alias
		# parse each files
		self._files = [PrFile(f, self, path, doParsing=parsing) for f in files]
		self.check_status()


	@property
	def types(self):
		"""Returns the list of types (keys)"""
		return list(self._types.keys())

	@property
	def variables(self):
		"""Returns the list of variables"""
		return self._variables.values()

	@property
	def typesDict(self):
		"""Returns the dictionary of types"""
		return self._types

	def setPerType(self):
		""""Returns a dictionary that, for each type, give the set of variables of that type
		so if _types is {0: 'double', 1: 'float', 2, 'double', 'x': 'float', 3: 'quad'}, setPerType will return
		{'double': {0, 2}, 'float': {1, 'x'}, 'quad': {3}}"""
		d = {}
		for k, v in self._types.items():
			if v in d:
				d[v] |= {k}
			else:
				d[v] = {k}
		return d

	def strResult(self):
		"""Returns the result (a string with each variable and its name)"""
		d = {}
		for k, v in self._types.items():
			var = ("%s (%s:%d)" % self._variables[k]) if k in self._variables else ("type " + str(k))
			if v in d:
				d[v] |= {var}
			else:
				d[v] = {var}
		return "\n".join("%s:\n%s\n" % (t, commaAnd(list(l))) for t, l in d.items())


	def registerVariable(self, name, typeName, fileName, lineNb):
		"""Register a variable with its type, filename and line number"""
		# if several variables share the same type, only the last one is now considered
		# (not a problem, this is only used for nice display...)
		self._variables[typeName] = (name, fileName, lineNb)


	@staticmethod
	def check_status():
		import os

		curr_loc = os.path.dirname(os.path.realpath(__file__))
		set_cadna_env = False

		if not os.path.isfile(curr_loc+'/cadna/lib/libcadnaC.a'):
			if 'CADNA_PATH' in os.environ:
				set_cadna_env = False
			else:
				import logging
				logging.basicConfig()
				log = logging.getLogger(__file__)

				log.warning(f"\nHave not found CADNA path." 
							f"\nPlease ensure CADNA is installed in this machine.")
				
		else:
			set_cadna_env = True
			
		if set_cadna_env:
			os.environ["CADNA_PATH"] = curr_loc+'/cadna/'
			# print("set cadna environment:", set_cadna_env, os.environ["CADNA_PATH"])
			# subprocess.call('export CADNA_PATH='+os.environ["CADNA_PATH"], shell=True)

	@property
	def compilations(self):
		"""Returns the nb of compilations and time of computations"""
		return self._compilations

	@property
	def executions(self):
		"""Returns the nb of executions and time of executions"""
		return self._executions

	@property
	def nbFiles(self):
		"""Returns the nb of files"""
		return len(self._files)

	@property
	def nbVariables(self):
		"""Returns the nb of variables"""
		return len(self._types)

	def expectation(self):
		"""Returns a string defining the expectation (nb of Digits)"""
		exp = "The expectation is %s digits" % self._nbDigits[0]
		if self._nbDigits[1]:
			exp += ' (and ' + commaAnd(['%d digits for variable `%s`' % (d, v) for v, d in self._nbDigits[1].items()]) + ')'
		return exp + '.'

	def registerType(self, typeName):
		"""when a __PROMISE__ or __PR_xxx__ type is found, we register it here
		(add its name in the dictionary _types) """
		if typeName:
			self._types[typeName] = None
			return typeName
		else:
			self._types[self._counter] = None
			self._counter += 1
			return self._counter - 1


	def changeSameType(self, Ctype):
		"""change the type for all the Types"""
		for k in self._types:
			self._types[k] = Ctype

	def changeSomeTypes(self, Ctype, lkeys):
		"""change the type of some keys"""
		for k in lkeys:
			self._types[k] = Ctype

	def changeTypes(self, dtype):
		"""change the type dictionary to a new dictionary"""
		for k in dtype:
			self._types[k] = dtype[k]

	def getTypesEqualTo(self, typ):
		"""returns the list of types (keys) equal to typ"""
		return [k for k, t in self._types.items() if t == typ]


	def compileAndRun(self, dest=None, cadna=False, result=True, compilationErrorAsDebug=False, compileErrorPath=None):
		"""
		- copy the project,
		- create the new files (from the original ones) if `createFiles` if True
		- compile it
		- run it
		- and get the result

		Parameters:
			- cadna: is True if Cadna is used (to create the reference)
			- result: True when we want the result (False otherwise)
			- compilationErrorasDebug: True when we just want to display the compilation error as debug
			- compileErrorCopy: (str) if not None, it copies the files in it if the compilation fails

		Returns True if the result is valid (with the nb of siginifcant digits greater thar required)"""

		logger.debug('runAndCompile with types=%s' % str(self._types))
		varFailed = None  # name of the variable that make the test failed

		# memoization
		if frozenset(self._types.items()) not in self._cache:
			# store actual directory and create destination folder (temporary or not)
			pwd = getcwd()
			if dest is None:
				dest = mkdtemp()
			else:
				runCommand(['mkdir', '-p', dest])

			# copy all the project (but exclude the dest/ in case it is in the same folder)
			runCommand(['rsync', '-a', join(self._path, '*'), dest, '--exclude', dest] + (['--exclude', compileErrorPath] if compileErrorPath else []))
			# copy all the useful files (promise.h, cadnaizer)
			runCommand(['cp', resource_filename(__name__, "extra/promise.h"), dest])
			runCommand(['cp', resource_filename(__name__, "extra/promise_header.h"), dest])
			runCommand(['cp', '-p', resource_filename(__name__, "extra/cadnaizer"), dest])

			# update the files that have to be changed
			for i, f in enumerate(self._files):
				f.createFile(self._types, dest, prefix='withoutcadna_' if cadna else '', promiseHeader=(i != 0))

			# move the  temp directory
			cd(dest)

			# cadnaize the project
			if cadna:
				for f in self._files:
					beginName, endName = split(f.fileName)
					filename = join(beginName, 'withoutcadna_' + endName)
					if not runCommand(['./cadnaizer', filename, '-o', f.fileName])[0]:
						cd(pwd)
						raise PromiseError("cadnaizer failed...")

			# compile it
			with Timing(self._compilations):
				for line in self._compile:
					res, msgError = runCommand(line.split(), errorsAsDegug=compilationErrorAsDebug, alias=self._alias)

					if not res:
						msg = "Compilation failed"
						if cadna:
							msg += "\nIs Cadna installed? Did you link your cade with Cadna with `-lcadnac`," \
							       " `-L$CADNA_PATH/lib` and `-I$CADNA_PATH/include` ?"
							if 'CADNA_PATH' not in environ:
								msg += '\nThe environment variable `CADNA_PATH` is not set, you should probably set it to Cadna path with'
								msg += '\n >>> export CADNA_PATH=/path/to/cadna'

						if compileErrorPath:
							# create compileError folder and copy all the files in it
							copyPath = join(pwd, compileErrorPath, 'attempt' + str(self.compilations[1]))
							runCommand(['mkdir', '-p', copyPath])
							runCommand(['cp', '-r', '.', copyPath])
							runFile = join(copyPath, "run.sh")
							# put all the compile commands in a `run.sh` file
							with open(runFile, 'w') as f:
								f.write("# Automatically generated by Promise (%s)\n\n" % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
								f.write('# Config=' + str(self.typesDict)+'\n\n')
								f.write("\n".join('#'+l for l in msgError))
								f.write("\n")
								f.write("\n".join(line for line in self._compile))
							#chmod(runFile, S_IXUSR)

						cd(pwd)
						raise PromiseCompilationError(msg)


			# run it and check if the execution has failed
			with Timing(self._executions):
				run = self._run.split(' ')
				ret, lines = runCommand(list(('./' if i == 0 else '') + r for i, r in enumerate(run)))

				if not ret:
					cd(pwd)
					raise PromiseError("Execution failed (should return 0)")

			# get the results in dictionary digits (and fill self._ref if cadna)
			if result and cadna:
				digits = {}
				self._ref = {}
				self._power = {}
				# parse the lines to get the variable name, 3 values and the nb of significant digits
				for li in lines:
					m = regDumpST.match(li)

					if m:
						varName, index, val1, val2, val3, nbDigits = m.groups()
						
						val = float.fromhex(val1)/3+float.fromhex(val2)/3+float.fromhex(val3)/3
						if varName not in digits:
							self._ref[varName] = []
							digits[varName] = []
						
						self._ref[varName].append(val)
						self._power[varName] = 10 ** (-self._nbDigits[1].get(varName, self._nbDigits[0]))
						digits[varName].append(int(nbDigits))

				# compare to expectations
				st = "\n".join("Variable %s: expect %d and found %d" % (var, self._nbDigits[1].get(var, self._nbDigits[0]), min(d))
				               for var, d in digits.items() if min(d) < self._nbDigits[1].get(var, self._nbDigits[0]))
				if st:
					cd(pwd)
					raise PromiseCompilationError(
						"The number of significant digits found with Cadna is lower than the expectation: ", st)

			elif result:
				# parse the result and store them in a dictionary
				values = {}
				for li in lines:
					m = regDump.match(li)

					if m:
						varName, index, val = m.groups()
						
						if varName not in values:
							values[varName] = []

						try:
							values[varName].append(float.fromhex(val))

						except ValueError as e:
							logger.error("ValueError val=%s\n%s" % (val, li))
							cd(pwd)
							raise ValueError from e
						
				# compare to the expectation
				try:
					if self._ref != {}:
						for var, d in values.items():
							for v, ref in zip(d, self._ref[var]):
								if isnan(v) or isinf(v):
									raise StopIteration(var)
								if abs(ref-v) >= abs(ref) * self._power[var]:
									if not((ref == 0) and (abs(v) <= self._power[var])):
										raise StopIteration(var)
								
				except StopIteration as e:
					varFailed = e.value


			# go back to previous folder
			cd(pwd)

			# store the result (memoization)
			if not cadna:
				self._cache[frozenset(self._types.items())] = (varFailed is None) if result else None
		else:
			logger.debug('\U00002192 Nothing to do (using result in cache)')

		# log the result
		if result:
			logger.debug('result: ' + ("\U0001F44D" if varFailed is None else
			                           ("\U0001F44E (because of variable " + varFailed + ")")))

		return self._cache[frozenset(self._types.items())] if not cadna else (varFailed is None) if result else None


	def exportFinalResult(self, path):
		"""Produces the final files in the path folder"""
		runCommand(['mkdir', '-p', path])
		tempDirList = []
		for f in self._files:
			# check if the file path contains extra directory which is not already created
			tempMiddleDirList = f.fileName.split('/')
			if len(tempMiddleDirList) > 1:
				# it contains extra directory
				tempMiddleDir = join(*tempMiddleDirList[0:-1])
				if tempMiddleDir not in tempDirList:
					# the extra directory is not already created, so create it
					tempDirList.append(tempMiddleDir)
					runCommand(['mkdir', '-p', join(path, tempMiddleDir)])
			f.createFile(self._types, path, final=True)


	def exportParsedFiles(self, path):
		"""Produces the files after the parsing"""
		# change the types equal to their name
		for k in self._types:
			if isinstance(k, str):
				self._types[k] = '__PR_' + k + '__'
			else:
				self._types[k] = '__PROMISE' + str(k) + '__'
		# produce the files
		runCommand(['mkdir', '-p', path])
		tempDirList = []
		for f in self._files:
			tempMiddleDirList = f.fileName.split('/')
			if len(tempMiddleDirList) > 1:
				# it contains extra directory
				tempMiddleDir = join(*tempMiddleDirList[0:-1])
				if tempMiddleDir not in tempDirList:
					# the extra directory is not already created, so create it
					tempDirList.append(tempMiddleDir)
					runCommand(['mkdir', '-p', join(path, tempMiddleDir)])
			f.createFile(self._types, path, final=True, prefix='parsed_')


	def runDeltaDebug(self, lowest, highest, tempPath, doPause=False, compileErrorPath=None):
		"""Run the Delta/Debug with lowest/highest precision
		first, try with lowest and then run DD algorithm
		return False if the DeltaDebug is not successful"""
		from tqdm import tqdm
		from math import ceil, log2
		import os
		n = len([v for v in self._types.values() if v == highest])
		nb = ceil(n * log2(n))

		from .deltadebug.dd import PromiseDD

		with tqdm(desc="Delta-Debug iterations", position=1, total=nb, leave=False, ncols=80,
				bar_format='%s{l_bar}%s{bar}%s| {n_fmt} it. [{rate_fmt}{postfix}]%s' %
							(Fore.LIGHTMAGENTA_EX, Fore.BLUE, Fore.LIGHTMAGENTA_EX, Fore.RESET)) as bar:
			with tqdm(total=0, position=0, 
					bar_format='%s{desc}%s' % (Fore.LIGHTMAGENTA_EX, Fore.RESET), leave=False) as status:
				ty = dict(self._types)
				self.changeSomeTypes(lowest, [k for k, v in self._types.items() if v == highest])
				try:
					if self.compileAndRun(tempPath, compilationErrorAsDebug=True):
						logger.info("The format %s is enough for your expectation", lowest)
						return True
					if doPause:
						pause()
				except PromiseCompilationError as err:
					logger.debug('Result: \U0001F44E (' + str(err) + ')')
				except PromiseError as err:
					logger.debug('Result: \U0001F44E (' + str(err) + ')')

				self._types = dict(ty)
				error_log_path = compileErrorPath
				if error_log_path and os.path.isdir(error_log_path):
					error_log_path = os.path.join(error_log_path, f"errors_{os.getpid()}.log")
				
				elif error_log_path is None:
					error_log_path = f"/tmp/promise_errors_{os.getpid()}.log"
				
				elif os.path.splitext(error_log_path)[1] == '.log' and os.path.isdir(os.path.splitext(error_log_path)[0]):
					error_log_path = os.path.join(os.path.splitext(error_log_path)[0], f"errors_{os.getpid()}.log")
				
				os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
				dd = PromiseDD(self, lowest, highest, status, bar, tempPath, doPause, error_log_path)
				dd.run()

				status.set_description_str('')
				status.refresh()
				bar.bar_format = '{desc}'
				bar.set_description_str('')
				bar.refresh()

		return len([v for v in self._types.values() if v == highest]) != n