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
Promise v3 has been written from v2 by Thibault Hilaire and Fabienne JÉZÉQUEL and Xinye Chen
	Promise v3 enables version check, pip installing, arbitrary precision, etc., features.
	Sorbonne Université
	LIP6 (Computing Science Laboratory)
	Paris, France. Contact: thibault.hilaire@lip6.fr, fabienne.jezequel@lip6.fr, xinyechenai@gmail.com

Some useful functions to parse the C/C++ files and replace the __PROMISE__ and __PR_xxx__ by some types.
To do that, we have to parse the file, in order to know if the __PROMISE__ are in a comment, string, etc. or
if it is used for a variable declaration, an array, a pointer, a function returns, a function argument, etc.
Depending on the case, the code may be a little bit changed
(`__PROMISE__ x,y=5;` is changed in `__PROMISE__ x; __PROMISE__ y; y=5;` for example)


© Thibault Hilaire and Fabienne JÉZÉQUEL, April 2024
"""

from .logger import PrLogger
from .errors import PromiseError
import regex as re
from itertools import zip_longest
from os.path import join, split, relpath

# logger
logger = PrLogger()

# WARNING: these regex (and the parsing done with them) is not robust to all the code
# if we want to be able to deal with these kind of cases (initialization part that is not just a litteral)
# we need to use a dedicated C parser (not home-made parser and few regex)

# regex to find the __PROMISE__, __PR_xxxx__ or __PRC_xxxx__
re_tag = re.compile('(.*?)(__PROMISE__|__PRC{0,1}_([^_]+)__)(.*)', re.DOTALL)

# regex for a variable
re_decl = re.compile(r'\s*(\**)\s*(\w+)((?:\[\w+\])*)\s*(.*)', re.DOTALL)
re_var_fun = re.compile(r'^(\**)\s*(\w+)(\[\w+\])*\s*(?:=\s*[\w\.]+)?')

# regex to remove the PROMISE_CHECK_VAR(...); and PROMISE_CHECK_ARRAY(...);
re_check = re.compile(r'PROMISE_CHECK_VAR\([\w]+\);|PROMISE_CHECK_ARRAY\([\w]+\);') # target


class VariableType:
	def __init__(self, key):
		self._key = key

	@property
	def value(self):
		return self._key


class CustomType(VariableType):
	pass


class NormalType(VariableType):
	pass


class Status:
	"""Enumeration for the status of a line
	(inside a /* .... */ comment, inside a //.... comment,
	inside a string "....." or not)"""
	NORMAL = 0
	COMMENT_C = 1
	COMMENT_CPP = 2
	STRING = 3
	PARENTHESIS = 4

	def __init__(self, status=None):
		"""create a empty stack for the status"""
		if status:
			self._stack = list(status.stack)
		else:
			self._stack = []

	def __eq__(self, other):
		"""compare to the last status"""
		if self._stack:
			return self._stack[-1] == other
		else:
			return Status.NORMAL == other

	def __ne__(self, other):
		"""negative compare"""
		return not (self == other)

	def __str__(self):
		d = {0: "NORMAL", 1: "COMMENT /*", 2: "COMMENT //", 3: "STRING", 4: "PARENTHESIS"}
		if not self._stack:
			return "NORMAL"
		else:
			return "[" + ", ".join(d[s] for s in self._stack) + "]"

	@property
	def stack(self):
		"""Returns the stack"""
		return self._stack

	def parseLine(self, line):
		"""Parse the line character by character to evalute the
		status at the line (see Status class), from the status before the line
		This is used to handle multi-line strings or multi-line comments"""
		# nothing to do for an empty line, otherwise, we parse it, char by char
		if line:
			prev = line[0]
			self._parseChar('', prev)
			for succ in line[1:]:
				# parse the two char to update the status stack
				change = self._parseChar(prev, succ)
				# if we have change the status, then c1 has been "consumed"
				# and should not be use for a new pattern (like in "/*/")
				prev = '' if change else succ


	def _parseChar(self, prev, cur):
		"""update the status stack according the current char (cur) and the previous char (prev)"""
		stsize = len(self._stack)
		# according to the status, check if we change the status
		if self == Status.NORMAL or self == Status.PARENTHESIS:
			# NORMAL status
			if prev == '/' and cur == '*':
				self._stack.append(Status.COMMENT_C)
			elif prev == '/' and cur == '/':
				self._stack.append(Status.COMMENT_CPP)
			elif cur == "\"":
				self._stack.append(Status.STRING)
			elif cur == "(":
				self._stack.append(Status.PARENTHESIS)
			# PARENTHESIS
			elif self == Status.PARENTHESIS and cur == ")":
				self._stack.pop()

		# are we inside a CPP comment // ?
		elif self == Status.COMMENT_CPP:
			if cur == '\n':
				self._stack.pop()
		# are we inside a C comment /* ?
		elif self == Status.COMMENT_C:
			if prev == '*' and cur == '/':
				self._stack.pop()
		# are we inside a string ?
		elif self == Status.STRING:
			if cur == "\"" and prev != "\\":
				self._stack.pop()

		# return True if something has changed
		return len(self._stack) != stsize


	def parseDecl(self, line):
		"""Generator that parses the declaration line
		(like 'x,y[24],*z, a = 25, b= toto(12,5);')
		returns (for each) the name of the variable, the code generated
		and then return the end of the line"""
		initPhase = False
		prev = line[0]
		deb = 0
		pos = 0
		for pos, cur in enumerate(line[1:], 2):
			if cur in [',', ')', ';'] and self == Status.NORMAL:
				# get the name of the variable, (pointer, array and init parts)
				decl = line[deb:pos-1]
				po, name, array, init = re_decl.match(decl).groups()
				# in case of initialization, separe declaration and initialization
				if initPhase:
					if array:
						decl = po + name + array + init + ";"
					else:
						decl = po + name + "; " + name + init + ";"
				else:
					decl += ';' if cur != ')' else ')'
				yield name, decl
				initPhase = False
				deb = pos
			# stop if we have ';' or '(' (not in initialization phase, ie *before* '=')
			if (cur == ';' or cur == ')') and self == Status.NORMAL:
				break
			elif cur == '(' and not initPhase and self == Status.NORMAL:
				# in that case, it is not a variable, but a function declaration
				decl = line[deb:pos]
				po, name, array, init = re_decl.match(decl).groups()
				yield name, decl
				break
			# if we have an '=', then we are after the initialization
			elif cur == '=' and self == Status.NORMAL:
				initPhase = True
			# parse the two char to update the status stack
			change = self._parseChar(prev, cur)
			prev = '' if change else cur
		else:
			if deb == 0:
				# it means that we have yield anything but we are at the end
				# (some special cases like __PROMISE__ x = (__PR__)xxx;)
				po, name, array, init = re_decl.match(line).groups()
				yield name, line

		# finally return the end of the line
		yield line[pos:]



class PrFile:
	"""PrFile class to handle the files
	a PrFile object correspond to a file that is treated by Promise
	the file is read, parsed and split at the __PROMISE__, __PR_xxx__ or __PRC_xxx__
	each of them correspond to a type that is handle by Promise (so transformed in quad, double, single or half)
	In a file, we keep a list of text and a list of indexes of C types
	To reconstruct the file, we just take the first part of text (first element in the list of text), then add the
	type corresponding to the first index , then the 2nd part of text, the 2nd types, etc.
	The types are taken from the dictionary of types (see the Promise object)
	"""

	_custom = {}    # a dictionary of custom types (used with __PRC_xxx__)

	def __init__(self, fileName, Pr, path, doParsing=False):
		"""Build the PrFile from the filename
		a Promise object is necessary (to register each __PROMISE__ or __PR_xxx__ type)"""
		self._fileName = fileName
		self._ltext = []        # list of strings
		self._typeKeys = []     # list of index of types
		self._path = path
		self._Pr = Pr

		# read and parse the file (split the file)
		try:
			with open(join(path, fileName), 'r') as file:
				if doParsing:
					self._complexConstruction(file, fileName)
				else:
					self._simpleConstruction(file)
		except FileNotFoundError:
			logger.error("the file %s doesn't exist !!", fileName)
			raise PromiseError("the file %s doesn't exist !!", fileName)


	@property
	def fileName(self):
		"""returns the fileName"""
		return self._fileName

	@classmethod
	def setCustom(cls, custom):
		"""Used to set the custom types
		(those used in __PRC_xxx__)"""
		cls._custom = custom

	def addKey(self, typeName, custom=False):
		"""add the typeName in the _typeKey list
		"""
		if custom:
			self._typeKeys.append(CustomType(typeName))
		else:
			self._typeKeys.append(NormalType(typeName))

	def treatVariableDeclaration(self, status, line, lineNb):
		"""this line is preceded by __PROMISE__ in the code
		this function replace a declaration like `__PROMISE x,y,z;` into multiple declarations
		like `__PROMISE__ x; __PROMISE__ y; __PROMISE__ z;`
		and treate them (add them to typeKeyps and ltext)
		Parameters
			-line: (string) line to consider
			- lineNb: (int) line number
		"""
		# function arguments or in a cast ?
		if self._ltext and status == Status.PARENTHESIS:
			# check if there is a variable within
			var_fun = re_var_fun.match(line)
			if var_fun:
				# register the variable
				self._Pr.registerVariable(var_fun.group(2), self._typeKeys[-1].value, self._fileName, lineNb)

		# a variable (or function) declaration
		elif self._ltext:
			# decompose the declarations/initializations (Ex 'x=12, y, z=5; blablabla')
			st = Status()
			res = list(st.parseDecl(line))
			variables = res[:-1]
			line = (' ' if isinstance(self._typeKeys[-1], NormalType) else '') + variables[-1][1].lstrip() + res[-1]
			for i, (name, varDecl) in enumerate(variables):
				# register the variable
				self._Pr.registerVariable(name, self._typeKeys[-1].value, self._fileName, lineNb)
				# add the type to text and typeKeys if it's not the  last one
				if i != len(variables)-1:
					self._ltext.append(' ' + varDecl.lstrip())
					# we use the same type as previously if it was a dedicated type (`__PR_xxx__`) and not `__PROMISE__`
					typeVar = self._Pr.registerType(None) if isinstance(self._typeKeys[-1].value, int) else self._typeKeys[-1].value
					self.addKey(typeVar, custom=isinstance(self._typeKeys[-1], CustomType))

		# at the end, we store the line
		self._ltext.append(line)

	def _simpleConstruction(self, file):
		"""Read the file and cut at __PROMISE__ or __PR_xxx__
		Old way to do, that do not consider strings, comment, mutliple variable declaration,
		declaration-and-initialization, etc."""
		endOfLine = ''
		for line in file:
			endOfLine = endOfLine + line
			mat = re_tag.match(endOfLine)
			while mat:
				self._ltext.append(mat.group(1))  # store the beginning of the line
				typeName = self._Pr.registerType(mat.group(3))  # get the key of the __PROMISE__ or __PR_xxx__ type
				self.addKey(typeName, mat.group(2).startswith('__PRC'))  # store the key
				endOfLine = mat.group(4)  # keep the end of the line
				mat = re_tag.match(endOfLine)  # match for the it
		self._ltext.append(endOfLine)

	def _complexConstruction(self, file, fileName):
		"""Read the file and do the parsing
		Srings, comments, mutliple variable declaration, declaration-and-initialization, etc.
		are considered"""
		lineNb = 0    # line number during the parsing
		status = Status()
		stack = []  # list of lines already parsed, and not yet added in self._ltext
		prevStatus = Status(status)
		for lineNb, line in enumerate(file):
			mat = re_tag.match(line)  # check for a __PROMISE__, __PR_xxx__ or __PRC_xxx__ inside line
			while mat:
				# parse the left part of the line and get the status
				status.parseLine(mat.group(1))
				if status == Status.NORMAL or status == Status.PARENTHESIS:
					if not mat.group(2).startswith('__PRC_'):
						# treat the 1st part (and the stack)
						self.treatVariableDeclaration(prevStatus, "".join(stack) + mat.group(1), lineNb)
					else:
						self._ltext.append("".join(stack) + mat.group(1))  # store the beginning of the line
					stack = []
					# get the key of the __PROMISE__ or __PR_xxx__ type
					typeName = self._Pr.registerType(mat.group(3))
					# store the key
					self.addKey(typeName, custom=mat.group(2).startswith('__PRC_'))
					prevStatus = Status(status)
				else:
					# we are in a comment or string
					# add the first part of the line and the __PROMISE__ in the stack
					stack.append(mat.group(1) + mat.group(2))
				# we continue on the rest of the line
				line = mat.group(4)
				mat = re_tag.match(line)
			# parse the line and add it the the stack
			status.parseLine(line)
			stack.append(line)
		# deal with the end of the file
		self.treatVariableDeclaration(prevStatus, "".join(stack), lineNb)

		# check the status at the end
		if status != Status.NORMAL:
			raise PromiseError("Something went wrong when parsing the file %s (status = %s)" % (fileName, status))


	def createFile(self, typeDict, path, prefix='', final=False, promiseHeader=False):
		"""create the file, where the __PROMISE__ and __PR_xxx__ are replaced by the vlaues in the `type` dictionary"""
		beginName, endName = split(self._fileName)
		filename = join(path, beginName, prefix + endName)
		logger.lowdebug("Create the file %s", filename)
		with open(filename, "w") as f:
			# include promise headers (except for the final version)
			if not final:
				f.write('#include <half_promise.hpp>\n')
				f.write('#include <floatx.hpp>\n')
				f.write('#include <fxmath.hpp>\n')
				relative = relpath(path='.', start=beginName)   # build the relative path from the file to the top folder (where promise.h is)
				incl = join(relative, "promise")
				f.write('#include "' + incl + ('_header.h"\n' if promiseHeader else '.h"\n'))
			
			else:
				f.write('#include <half.hpp>\n')
				f.write('#include <floatx.hpp>\n')
				
			for text, key in zip_longest(self._ltext, self._typeKeys):
				if final:
					f.write(re.sub(re_check, '', text))
				else:
					f.write(text)
				if key is not None:
					if isinstance(key, NormalType):
						f.write(typeDict[key.value])
					elif isinstance(key, CustomType):
						f.write(self._custom.get(typeDict[key.value], typeDict[key.value]))   # The CustomDict is skiped if the typeDict[] is not in the dictionary


