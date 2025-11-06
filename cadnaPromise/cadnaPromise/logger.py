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

	Some useful functions for the logger
	Mainly, each file instanciates a PrLogger (that contains an instance of a logger)
	At the beginning, the logger is configured (from the yaml file and the command line options)

	Two handlers handle the logs:
	- one for display the information to the terminal
	- one for the log file
	The level set depends on the verbosity (--verbosity and --verbosityLog options). There are 5 levels of verbosity

	List of logging level, and the verbosity level (for each level, display everything above):
	- (40) ERROR (error messages)
	- (25) MESSAGE (start up, end messages)             <- verbosity 0
	- (20) INFO (steps done)                            <- verbosity 1
	- (10) DEBUG (some debug info)                      <- verbosity 2
	- (7) OUTPUT (output of the runs)
	- (6) COMMAND (command that are done)               <- verbosity 3
	- (5) LOW_DEBUG (low level debug messages)
	- (3) OUTPUT_PROMISE (values display by promise.h)  <- verbosity 4



	© Thibault HILAIRE, April 2020
"""


from colorlog import ColoredFormatter  # logging with colors
import logging
from os.path import join, expanduser
from .errors import PromiseError
from tqdm import tqdm


# logging level (see logging module) and their color, with respect to the verbosity
levels = [25, 20, 10, 6, 3]
log_colors = {
	'ERROR': 'red', 'MESSAGE': 'blue', 'INFO': 'white', 'OUTPUT': 'cyan,', 'DEBUG': 'green',
	'COMMAND': 'purple', 'LOW_DEBUG': 'yellow', 'OUTPUT_PROMISE': 'cyan'
}


class TqdmLoggingHandler(logging.Handler):
	"""	A dedicated Logging Handler
	to be used with tqdm progress bar, see
https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
	for more informationns"""
	def __init__(self, level=logging.NOTSET):
		"""Set up the super logger"""
		super().__init__(level)

	def emit(self, record):
		"""emit method, display message using tqdm.write instead of print"""
		msg = self.format(record)
		tqdm.write(msg)
		self.flush()



class PrLogger:
	"""Dedicated logger for Promise
	dedicated levels, dedicated handlers that are removed at the end"""

	def __init__(self):
		"""just get the root logger"""
		self._logger = logging.getLogger('promise')
		self._step = 'a'  # step

	def reset(self):
		"""Reset the logger"""
		self._step = 'a'
		for h in self._logger.handlers[:]:
			h.close()
			self._logger.removeHandler(h)

		self._logger.handlers.clear() # add by Xinye

	# -- log levels --
	def error(self, msg, *args, **kws):
		"""log at message at level 40 (error)"""
		self._logger.log(40, msg, *args, **kws)

	def message(self, msg, *args, **kws):
		"""log at message at level 25 (message)"""
		self._logger.log(25, msg, *args, **kws)

	def info(self, msg, *args, **kws):
		"""log at message at level 20 (info)"""
		self._logger.log(20, msg, *args, **kws)

	def debug(self, msg, *args, **kws):
		"""log at message at level 10 (debug)"""
		self._logger.log(10, msg, *args, **kws)

	def output(self, msg, *args, **kws):
		"""log at message at level 5 (low level debug)"""
		self._logger.log(7, msg, *args, **kws)

	def command(self, msg, *args, **kws):
		"""log at message at level 5 (low level debug)"""
		self._logger.log(6, msg, *args, **kws)

	def lowdebug(self, msg, *args, **kws):
		"""log at message at level 5 (low level debug)"""
		self._logger.log(5, msg, *args, **kws)

	def outputPromise(self, msg, *args, **kws):
		"""log at message at level 5 (very low level debug)"""
		self._logger.log(3, msg, *args, **kws)

	def step(self, string):
		"""log a new step"""
		self._logger.info(self._step + ') ' + string)
		self._step = chr(ord(self._step) + 1)


	def configureLogger(self, options):
		"""Configure the logger from the options of the *command line* (only)
		(not from the configuration file, because we need to configure the logger
		before reading the configuration file, otherwise we cannot log the potential errors
		while parsing it)
		"""
		# get the path
		path = options['--path'] if options['--path'] else ''
		# get the verbosity
		verbosity = getInt(self._logger, options, 'verbosity', 1, 4, default=1)
		verbosityLog = getInt(self._logger, options, 'verbosityLog', 1, 4, default=verbosity)
		options['--verbosityLog'] = verbosityLog
		# set new logger levels for messages and low level debug
		logging.addLevelName(25, "MESSAGE")
		logging.addLevelName(7, "OUTPUT")
		logging.addLevelName(6, "COMMAND")
		logging.addLevelName(5, "LOW_DEBUG")
		logging.addLevelName(3, "OUTPUT_PROMISE")
		# stream the log to the console, with colors
		stream_handler = TqdmLoggingHandler()
		stream_handler.setLevel(levels[verbosity])
		stream_handler.setFormatter(ColoredFormatter("%(log_color)s%(message)s%(reset)s", log_colors=log_colors))
		self._logger.addHandler(stream_handler)
		# stream the log to the file
		if options['--log']:
			self.addFileLogger(path, options['--log'], verbosityLog)
		# set the logger to the minimum level (the handler will filter the messages)
		self._logger.setLevel(3)


	def addFileLogger(self, path, filename, verbosity):
		"""Add a file handler for the logger
		it is be called from `configureLogger` (when the command line options is parsed)
		and also by `getYMLoptions` when the yml file is parsed
		(the file logger should be configure *before* parsing the yml file,
		in order to log the potential errors of the parsing)"""
		file_handler = logging.FileHandler(expanduser(join(path, filename)), mode='w')
		file_handler.setLevel(levels[verbosity])
		error_formatter = logging.Formatter('%(asctime)s: %(message)s', "%m/%d %H:%M:%S")
		file_handler.setFormatter(error_formatter)
		self._logger.addHandler(file_handler)



def getInt(logger, options, name, mini, maxi, default=None):
	"""get an integer in the options, betwen min and max
	if not present, the default value is taken
	use the logger to log the errors, if happens"""
	optName = '--' + name
	try:
		val = int(options[optName]) if (optName in options) and (options[optName]) else default
	except ValueError as err:
		logger.error('The ' + name + ' is incorrect (`--' + name + '` option)')
		raise PromiseError(err) from err
	if not (mini <= val <= maxi):
		logger.error('The %s should be between %d and %d (included) (`--%s` option)' % (name, mini, maxi, name))
		raise PromiseError
	return val



