# coding: utf-8
 
from setuptools import setup


def readme():
	"""import the readme file"""
	with open('README.rst') as f:
		return f.read()

def get_version(fname):
    with open(fname) as f:
        for line in f:
            if line.startswith("__version__ = '"):
                return line.split("'")[1]
    raise RuntimeError('Error in parsing version string.')


setup_args = {'name': 'cadnaPromise',
		'version':  '0.3.3',
		'description':'Precision auto-tuning of floating-point variables in program',
		'long_description':readme(),
		'classifiers':["Intended Audience :: Science/Research",
					"Intended Audience :: Developers",
					"Programming Language :: C",
					"Programming Language :: Python",
					"Topic :: Software Development",
					"Topic :: Scientific/Engineering",
					'Operating System :: POSIX',
					'Operating System :: Unix',
					'Operating System :: MacOS',
					"Programming Language :: Python :: 3",
					],
		'keywords':'computer arithmetic mixed-precision, precision auto-tuning',
		'url':'https://github.com/PEQUAN/cadnaPromise',
		'author':'LIP6 PEQUAN team',
		'author_email':'thibault.hilaire@lip6.fr; fabienne.jezequel@lip6.fr',
		'license':'GNU General Public License v3.0',
		'packages':{"cadnaPromise",
			  		"cadnaPromise.cadna", 
					"cadnaPromise.deltadebug",  
					"cadnaPromise.extra",
					"cadnaPromise.cache"
					},
		'package_data':{"cadnaPromise": ["deltadebug/*", "cadna/*", "extra/*", "cache/*"]},
		'tests_require':['pytest', 'pytest-cov'],
		'setup_requires':['colorlog', 'colorama', 'regex', 'setuptools'],
		'install_requires':['colorlog', 'colorama', 'tqdm', 'regex', 'pyyaml', 'docopt-ng'],
		'extras_require':{'with_doc': ['sphinx', 'sphinx_bootstrap_theme']},
		'include_package_data':True,
		'data_files':[('extra/', ['cadnaPromise/extra/promise.h', 'cadnaPromise/extra/cadnaizer'])],
		'zip_safe':False,
		'entry_points':{'console_scripts': 
						['promise=cadnaPromise.run:runPromise',
						 'activate-promise=cadnaPromise.install:activate', 
                         'deactivate-promise=cadnaPromise.install:deactivate',
                         'load_CADNA_PATH=cadnaPromise.run:loadCADNA',
                         'promise-batch=cadnaPromise.run:run_experiment_and_plot',
                        ]
		}}

setup(**setup_args)
