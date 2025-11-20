# this file was originally written by Zeller for Delta Debug
# see the original credtis (modifications logged at the end)

# $Id: DD.py,v 1.2 2001/11/05 19:53:33 zeller Exp $
# Enhanced Delta Debugging class
# Copyright (c) 1999, 2000, 2001 Andreas Zeller.

# This module (written in Python) implements the base delta debugging
# algorithms and is at the core of all our experiments.  This should
# easily run on any platform and any Python version since 1.6.
#
# To plug this into your system, all you have to do is to create a
# subclass with a dedicated `test()' method.  Basically, you would
# invoke the DD test case minimization algorithm (= the `ddmin()'
# method) with a list of characters; the `test()' method would combine
# them to a document and run the test.  This should be easy to realize
# and give you some good starting results; the file includes a simple
# sample application.
#
# This file is in the public domain; feel free to copy, modify, use
# and distribute this software as you wish - with one exception.
# Passau University has filed a patent for the use of delta debugging
# on program states (A. Zeller: `Isolating cause-effect chains',
# Saarland University, 2001).  The fact that this file is publicly
# available does not imply that I or anyone else grants you any rights
# related to this patent.
#
# The use of Delta Debugging to isolate failure-inducing code changes
# (A. Zeller: `Yesterday, my program worked', ESEC/FSE 1999) or to
# simplify failure-inducing input (R. Hildebrandt, A. Zeller:
# `Simplifying failure-inducing input', ISSTA 2000) is, as far as I
# know, not covered by any patent, nor will it ever be.  If you use
# this software in any way, I'd appreciate if you include a citation
# such as `This software uses the delta debugging algorithm as
# described in (insert one of the papers above)'.
#
# All about Delta Debugging is found at the delta debugging web site,
#
#               http://www.st.cs.uni-sb.de/dd/
#
# Happy debugging,
#
# Andreas Zeller


# Copyright 2016 Romain PICOT
#
# This file is part of PROMISE.
#
#    PROMISE is free software: you can redistribute it and/or modify it
#    under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    PROMISE is distributed in the hope that it will be useful, but WITHOUT
#    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General
#    Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with PROMISE. If not, see
#    <http://www.gnu.org/licenses/>.
#


# Modification by Romain PICOT:
# Patch to use the Delta-Debug algorithm from
# "WHY PROGRAMS FAIL: A Guide to Systematic Debugging"

# Modification by Thibault HILAIRE (Sorbonne Université, LIP6, 2020)
# - port to Python3 (with 2to3), restore tabs, comply to PEP8
# - add (custom) logger
# - rewrite some parts in a more pythonic way


# TODO: remove unnecessary functions, tests, etc. - finished by Xinye Chen, xinyechenai@gmail.com
# TODO: remove the use of self.debug_xxxx (now dealed with the logger)

# get the logger
from ..logger import PrLogger


from ..errors import PromiseCompilationError
from ..utils import pause
from ..errors import PromiseError



logger = PrLogger()



# Start with some helpers.
class OutcomeCache:
	"""This class holds test outcomes for configurations.  This avoids
	running the same test twice.

	The outcome cache is implemented as a tree.  Each node points
	to the outcome of the remaining list.

	Example: ([1, 2, 3], PASS), ([1, 2], FAIL), ([1, 4, 5], FAIL):

		 (2, FAIL)--(3, PASS)
		/
	(1, None)
		\
		 (4, None)--(5, FAIL)"""

	def __init__(self):
		self.tail = {}                  # Points to outcome of tail
		self.result = None              # Result so far

	def add(self, C, result):
		"""Add (C, RESULT) to the cache.  C must be a list of scalars."""
		cs = C[:]
		cs.sort()

		p = self
		for start in range(len(C)):
			if C[start] not in p.tail:
				p.tail[C[start]] = OutcomeCache()
			p = p.tail[C[start]]

		p.result = result

	def lookup(self, C):
		"""Return RESULT if (C, RESULT) is in the cache; None, otherwise."""
		p = self
		for start in range(len(C)):
			if C[start] not in p.tail:
				return None
			p = p.tail[C[start]]

		return p.result

	def lookup_superset(self, C, start=0):
		"""Return RESULT if there is some (C', RESULT) in the cache with
		C' being a superset of C or equal to C.  Otherwise, return None."""

		# FIXME: Make this non-recursive!
		if start >= len(C):
			if self.result:
				return self.result
			elif self.tail != {}:
				# Select some superset
				superset = self.tail[list(self.tail.keys())[0]]
				return superset.lookup_superset(C, start + 1)
			else:
				return None

		if C[start] in self.tail:
			return self.tail[C[start]].lookup_superset(C, start + 1)

		# Let K0 be the largest element in TAIL such that K0 <= C[START]
		k0 = None
		for k in list(self.tail.keys()):
			if (k0 is None or k > k0) and k <= C[start]:
				k0 = k

		if k0 is not None:
			return self.tail[k0].lookup_superset(C, start)

		return None

	def lookup_subset(self, C):
		"""Return RESULT if there is some (C', RESULT) in the cache with
		C' being a subset of C or equal to C.  Otherwise, return None."""
		p = self
		for start in range(len(C)):
			if C[start] in p.tail:
				p = p.tail[C[start]]

		return p.result


# helpers
def listminus(C1, C2):
	"""Return a list of all elements of C1 that are not in C2."""
	s2 = {delta: 1 for delta in C2}
	C = [delta for delta in C1 if delta not in s2]
	return C


def listintersect(C1, C2):
	"""Return the common elements of C1 and C2."""
	s2 = {delta: 1 for delta in C2}
	C = [delta for delta in C1 if delta in s2]
	return C


def listunion(C1, C2):
	"""Return the union of C1 and C2."""
	s1 = {delta: 1 for delta in C1}
	C = C1[:] + [delta for delta in C2 if delta not in s1]
	return C


def listsubseteq(C1, C2):
	"""Return 1 if C1 is a subset or equal to C2."""
	s2 = {delta: 1 for delta in C2}

	for delta in C1:
		if delta not in s2:
			return 0

	return 1


# Main Delta Debugging algorithm.
class DD:
	"""Delta debugging base class.  To use this class for a particular
	setting, create a subclass with an overloaded `test()' method.

	Main entry points are:
	- `ddmin()' which computes a minimal failure-inducing configuration, and
	- `dd()' which computes a minimal failure-inducing difference.

	See also the usage sample at the end of this file.

	For further fine-tuning, you can implement an own `resolve()'
	method (tries to add or remove configuration elements in case of
	inconsistencies), or implement an own `split()' method, which
	allows you to split configurations according to your own
	criteria.

	The class includes other previous delta debugging alorithms,
	which are obsolete now; they are only included for comparison
	purposes."""

	# Test outcomes.
	PASS = "PASS"
	FAIL = "FAIL"
	UNRESOLVED = "UNRESOLVED"

	# Resolving directions.
	ADD = "ADD"                        # Add deltas to resolve
	REMOVE = "REMOVE"                        # Remove deltas to resolve

	# Debugging output (set to 1 to enable)
	debug_test = 0
	debug_dd = 0
	debug_split = 0
	debug_resolve = 0

	def __init__(self):
		self.__resolving = 0
		self.__last_reported_length = 0
		self.monotony = 0
		self.outcome_cache = OutcomeCache()
		self.cache_outcomes = 1
		self.minimize = 1
		self.maximize = 1
		self.assume_axioms_hold = 1

	# Output
	def coerce(self, C):
		"""Return the configuration C as a compact string"""
		# Default: use printable representation
		return repr(C)

	def pretty(self, C):
		"""Like coerce(), but sort beforehand"""
		sorted_c = C[:]
		sorted_c.sort()
		return self.coerce(sorted_c)

	# Testing
	def test(self, C):
		"""Test the configuration C.  Return PASS, FAIL, or UNRESOLVED"""
		C.sort()

		# If we had this test before, return its result
		if self.cache_outcomes:
			cached_result = self.outcome_cache.lookup(C)
			if cached_result is not None:
				return cached_result

		if self.monotony:
			# Check whether we had a passing superset of this test before
			cached_result = self.outcome_cache.lookup_superset(C)
			if cached_result == self.PASS:
				return self.PASS

			cached_result = self.outcome_cache.lookup_subset(C)
			if cached_result == self.FAIL:
				return self.FAIL

		if self.debug_test:
			logger.lowdebug("test(" + self.coerce(C) + ")...")

		outcome = self._test(C)

		if self.debug_test:
			logger.lowdebug("test(" + self.coerce(C) + ") = " + repr(outcome))

		if self.cache_outcomes:
			self.outcome_cache.add(C, outcome)

		return outcome

	def _test(self, C):
		"""Stub to overload in subclasses"""
		return self.UNRESOLVED                # Placeholder

	
	def split(self, C, n): # Splitting
		"""Split C into [C_1, C_2, ..., C_n]."""
		if self.debug_split:
			logger.lowdebug("split(" + self.coerce(C) + ", " + repr(n) + ")...")

		outcome = self._split(C, n)

		if self.debug_split:
			logger.lowdebug("split(" + self.coerce(C) + ", " + repr(n) + ") = " + repr(outcome))

		return outcome

	def _split(self, C, n):
		"""Stub to overload in subclasses"""
		subsets = []
		start = 0
		for i in range(n):
			subset = C[start:start + (len(C) - start) // (n - i)]
			subsets.append(subset)
			start = start + len(subset)
		return subsets

	def resolve(self, csub, C, direction):# Resolving
		"""If direction == ADD, resolve inconsistency by adding deltas
			 to CSUB.  Otherwise, resolve by removing deltas from CSUB."""

		if self.debug_resolve:
			logger.lowdebug("resolve(" + repr(csub) + ", " + self.coerce(C) + ", " + repr(direction) + ")...")

		outcome = self._resolve(csub, C, direction)

		if self.debug_resolve:
			logger.lowdebug("resolve(" + repr(csub) + ", " + self.coerce(C) + ", " + repr(direction) + ") = " + repr(outcome))

		return outcome


	def _resolve(self, csub, C, direction):
		"""Stub to overload in subclasses."""
		# By default, no way to resolve
		return None

	def test_and_resolve(self, csub, r, C, direction): # Test with fixes
		"""Repeat testing CSUB + R while unresolved."""

		initial_csub = csub[:]
		C2 = listunion(r, C)

		csubr = listunion(csub, r)
		t = self.test(csubr)

		# necessary to use more resolving mechanisms which can reverse each
		# other, can (but needn't) be used in subclasses
		self._resolve_type = 0

		while t == self.UNRESOLVED:
			self.__resolving = 1
			csubr = self.resolve(csubr, C, direction)

			if csubr is None:
				# Nothing left to resolve
				break
			else:
				if len(csubr) >= len(C2):
					# Added everything: csub == c2. ("Upper" Baseline)
					# This has already been tested.
					csubr = None
					break

				if len(csubr) <= len(r):
					# Removed everything: csub == r. (Baseline)
					# This has already been tested.
					csubr = None
					break

				t = self.test(csubr)

		self.__resolving = 0
		if csubr is None:
			return self.UNRESOLVED, initial_csub

		assert t == self.PASS or t == self.FAIL
		csub = listminus(csubr, r)
		return t, csub

	def resolving(self): # Inquiries
		"""Return 1 while resolving."""
		return self.__resolving

	def report_progress(self, C, title): # Logging
		if len(C) != self.__last_reported_length:
			logger.lowdebug(title + ": " + repr(len(C)) + " deltas left")
			self.__last_reported_length = len(C)

	def test_mix(self, csub, C, direction): # Delta Debugging (old ESEC/FSE version)
		t = self.FAIL   # default value, never used
		if self.minimize:
			t, csub = self.test_and_resolve(csub, [], C, direction)
			if t == self.FAIL:
				return t, csub

		if self.maximize:
			csubbar = listminus(self.CC, csub)
			cbar = listminus(self.CC, C)
			if direction == self.ADD:
				directionbar = self.REMOVE
			else:
				directionbar = self.ADD

			(tbar, csubbar) = self.test_and_resolve(csubbar, [], cbar, directionbar)

			csub = listminus(self.CC, csubbar)

			if tbar == self.PASS:
				t = self.FAIL
			elif tbar == self.FAIL:
				t = self.PASS
			else:
				t = self.UNRESOLVED

		return t, csub


	# Delta Debugging (new ISSTA version)
	def ddmax(self, C):
		return self.ddgen(C, 0, 1)

	def ddmin(self, C):
		return self.ddgen(C, 1, 0)

	def ddmix(self, C):
		return self.ddgen(C, 1, 1)


	def ddgen(self, C, minimize, maximize):
		"""Return a 1-minimal failing subset of C"""

		self.minimize = minimize
		self.maximize = maximize

		n = 2
		self.CC = C

		if self.debug_dd:
			logger.lowdebug(("dd(" + self.pretty(C) + ", " + repr(n) + ")..."))

		outcome = self._dd(C, n)

		if self.debug_dd:
			logger.lowdebug(("dd(" + self.pretty(C) + ", " + repr(n) + ") = " + repr(outcome)))

		return outcome

	def _dd(self, C, n):
		"""Stub to overload in subclasses"""

		# assert self.test([]) == self.PASS
		run = 1
		cbar_offset = 0

		# We replace the tail recursion from the paper by a loop
		while 1:
			# tc = self.test(c)
			# assert tc == self.FAIL or tc == self.UNRESOLVED
			if n > len(C):
				# No further minimizing
				logger.lowdebug("dd: done")
				return C

			self.report_progress(C, "dd")

			cs = self.split(C, n)

			logger.debug("dd (run #" + repr(run) + "): trying " + "+".join(str(len(cc)) for cc in cs))

			c_failed = 0
			cbar_failed = 0

			next_c = C[:]
			next_n = n

			if not c_failed:
				# Check complements
				cbars = n * [self.UNRESOLVED]

				# logger.debug "cbar_offset =", cbar_offset

				for j in range(n):
					if cbar_offset == 0:
						i = (j + cbar_offset) % (n + 1)
					else:
						i = (j + cbar_offset) % n
					cbars[i] = listminus(C, cs[i])
					t, cbars[i] = self.test_mix(cbars[i], C, self.ADD)

					doubled = listintersect(cbars[i], cs[i])
					if doubled is not []:
						cs[i] = listminus(cs[i], doubled)

					if t == self.FAIL:
						if self.debug_dd:
							logger.lowdebug("dd: reduced to" + str(len(cbars[i])) + "deltas:" + self.pretty(cbars[i]))

						cbar_failed = 1
						next_c = listintersect(next_c, cbars[i])
						next_n = next_n - 1
						self.report_progress(next_c, "dd")

						# In next run, start removing the following subset
						cbar_offset = i
						break

			if not c_failed and not cbar_failed:
				if n >= len(C):
					# No further minimizing
					logger.lowdebug("dd: done")
					return C

				next_n = min(len(C), n * 2)
				logger.lowdebug("dd: increase granularity to" + str(next_n))
				cbar_offset = (cbar_offset * next_n) // n

			C = next_c
			n = next_n
			run = run + 1


	# General delta debugging (new TSE version)
	def dd(self, C):
		return self.dddiff(C)           # Backwards compatibility




class PromiseDD(DD):
	"""Promise Delta Debug"""

	def __init__(self, pr, type1, type2, status, bar, path='', doPause=False, compileErrorPath=None):
		"""Create a Delta Debug object, with promise-dedicated test method
		Parameter:
		- pr: the Promise object of the project
		- type1, type2: (string) the two types uses for the delta-debug method
		(all the types initialy equal to types2 can be changed to type1)
		- status, bar: two tqdm objects (status and progress bar)
		- path: (string) path where put the generated files (compiled, run and tested)
			if empty, a temporary folder is used
		- doPause: (boolean) True if we pause after each dd iteratio"""
		self._status = status
		self._bar = bar
		self._type1 = type1
		self._type2 = type2
		self._pr = pr
		self._path = path
		self._pause = doPause
		self._initialTypes = dict(pr.typesDict)
		self._compileErrorPath = compileErrorPath
		super().__init__()

	def test(self, C):
		"""Test if the configuration
		Returns PASS, FAIL, or UNRESOLVED."""
		logger.debug('We test C=' + str(C))
		# change the types according to C
		self._pr.changeTypes(self._initialTypes)
		self._pr.changeSomeTypes(self._type1, C)
		# compile, run and get the result
		try:
			check = self._pr.compileAndRun(self._path, compilationErrorAsDebug=True, compileErrorPath=self._compileErrorPath)
			if self._pause:
				pause(self._status)
		except PromiseCompilationError:
			logger.debug('Compilation failed \U0001F44E')
			self._status.set_description_str('Compilation failed \U0001F44E')
			self._status.refresh()

			return DD.FAIL
		
		except PromiseError as err:
			logger.debug('Result failed: \U0001F44E (' + str(err) + ')')
			return DD.FAIL
		
		# update the status
		self._status.set_description_str('We test C=' + str(C) + (" \U0001F44D" if check else " \U0001F44E"))
		self._bar.update()
		return DD.PASS if check else DD.FAIL


	def run(self):
		"""run the Delta Debug algorithm
		Retuns what ddmax returns"""
		# run the dd
		deltas = self._pr.getTypesEqualTo(self._type2)
		t = self.ddmax(deltas)
		# set progress bar to 100%
		self._bar.total = self._bar.n
		self._bar.refresh()
		# return to the types given by dd algorithm
		self._pr.changeTypes(self._initialTypes)
		self._pr.changeSomeTypes(self._type1, list(set(deltas)-set(t)))
		# print("DD algorithms returns : " + str(self._pr.typesDict))
