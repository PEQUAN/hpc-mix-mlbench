# coding=utf-8

""" Promise v3 tests with the examples"""


from pytest import mark
#from tempfile import mkdtemp
#from os.path import join
#from cadnaPromise import Promise
from cadnaPromise.run import runPromise
#from cadnaPromise.prfile import PrFile


# list of examples and their expected result
examples = [
	['arclength1', 'sd', '4', {'d': {'fun', 3}, 's': {0, 1, 2, 'd1', '1'}}],
	['arclength1', 'hsd', '3', {'s': {0, 1, 3, '1', 'fun'}, 'h': {'d1', 2}}],
	['arclength1', 'hsd', '1', {'s': {0, 1, 3, '1', 'fun'}, 'h': {'d1', 2}}],
	['arclength1', 'hsd', '6', {'d': {0, 'fun', 1, 3, '1'}, 's': {2}, 'h': {'d1'}}],
	['arclength2', 'sd', '4', {'d': {0, 'fun'}, 's': {1, '1', 'd1'}}],
	['arclength2', 'hsd', '3', {'s': {0, '1', 'fun'}, 'h': {'d1', 1}}],
	['arclength2', 'hsd', '1', {'s': {0, 'fun', '1'}, 'h': {1, 'd1'}}],
	['arclength2', 'hsd', '6', {'d': {0, 'fun', '1'}, 's': {1}, 'h': {'d1'}}],
	['Heattransfer', 'sd', 8, {'s': {1, 2}, 'd': {0, '1'}}],
	['rectangleMethod', 'sd', 10, {'d': {1, 3, 4, 5}, 's': {0, 2}}],
	['squareRoot', 'sd', 8, {'d': {0, 2, 3, 4, 6, 7}, 's': {1, 5}}],
	['SP0', 'hsd', 3, {
		's': {2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 27},
		'h': {0, 1, 10, 11, 12, 13, 24, 25, 26}}],
	['SP1', 'sd', 12, {
		's': {1, 24},
		'd': {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25}}],
	['SP2', 'hs', 2, {'h': {0, 1, 3, 4, 7, 8, 10, 11}, 's': {2, 5, 6, 9, 12}}],
	['SP3', 'hsd', 3, {'h': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}}],
	['SP4', 'sd', 8, {'d': {0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20}, 's': {8, 10, 12, 6}}],
	['SP5', 'sd', 8, {
		'd': {0, 2, 3, 4, 35, 37, 7, 9, 11, 13, 15, 17, 19, 21, 23, 27, 29, 31},
		's': {1, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 26, 28, 30, 32, 33, 34, 36}}],
	['SP6', 'cd', 12, {
		'd': {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25},
		'c': {24, 1}}], # to test one customized floating point types
	['SP7', 'cwd', 3, {
		'c': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}}], # following SP3, to test two customized floating point types
	['SP7', 'hwd', 3, {
		'h': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}}], # following SP3, to test two customized floating point types
	['SP7', 'hsd', 15, {
		'd': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}}] # following SP3, to test two customized floating point types
]


types = {'c':'flx::floatx<5, 10>', 'w': 'flx::floatx<8, 23>', 'h': 'half_float::half', 's': 'float', 'd': 'double', 'q': 'float128'}

global alias
alias = ""

def pytest_generate_tests(metafunc):
	""" just to attach the cmd-line args to a test-class that needs them """
	global alias
	alias = metafunc.config.getoption("alias")
	# print("alias 1:", alias)
	print('command line passed for --alias ({})'.format(alias))


@mark.parametrize("name, method, digits, res", examples)
def test_examples(name, method, digits, res):
	"""Test the examples in the examples/ folder
	run Promise on these tests, and then compare with the expectation"""
	if name in ['SP6', 'SP7']:
		testargs = [
			'--precs='+method, '--nbDigits=' + str(digits), '--conf=%s/promise.yml' % name, '--fp=%s/fp.json' % name, 
			'--path=%s' % name, '--log=', '--verbosityLog=4', '--parsing', '--verbosity=1']
	else:
		testargs = [
			'--precs='+method, '--nbDigits=' + str(digits), '--conf=%s/promise.yml' % name, 
			'--path=%s' % name, '--log=', '--verbosityLog=4', '--parsing', '--verbosity=1']

	# add the alias parameter if existing
	if alias:
		# print("alias 2:", alias) # 
		testargs.append(" ".join("--alias="+t for t in alias))

	# rune Promise and check if the result is as expected
	t = runPromise(testargs)
	assert t == {types[k]: v for k, v in res.items()}

