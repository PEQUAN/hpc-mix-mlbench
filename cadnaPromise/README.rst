cadnaPromise
==============

.. image:: https://img.shields.io/badge/License-GPLv3-yellowgreen.svg
    :target: LICENSE
    :alt: License


.. image:: https://gitlab.lip6.fr/hilaire/promise2/badges/master/pipeline.svg
    :target: pipeline
    :alt: Pipeline
---- 

``cadnaPromise`` is a precision auto-tuning software using command-line interfaces.


--------
Install
--------

To install ``cadnaPromise``, simply use the pip command in terminal:  

.. parsed-literal::

  pip install cadnaPromise


after that, to enable the arbitrary precision customization and cadna installation, simply activate the ``cadnaPromise`` via the command in terminal:

.. parsed-literal::

  activate-promise


To reverse the process, simply do 

.. parsed-literal::

  deactivate-promise


Besides, users can install ``floatx`` and ``CADNA`` outside, and then specifying the path via

.. parsed-literal::

	export CADNA_PATH=[YOURPATH]





Check the if ``cadnaPromise`` is installed:

.. parsed-literal::

  promise --version


-----------------------
Usage and configuration
-----------------------


The installation of ``cadnaPromise`` requires the the following Python libraries: ``colorlog``, ``colorama``, ``pyyaml``, ``regex``.

The compiling of ``cadnaPromise`` requires ``g++``. Please ensure the installation of above libraries for a proper running of cadnaPromise.

Before the run of PROMISE, we require the following configration for the code to be tested:

Setting up ``promise.yml`` file
--------------------------------


Before running the program, users can customize the configuration file ``promise.yml``. 

.. parsed-literal::

	compile:
	- g++ [SOURCE FILES] -frounding-math -m64 -o [OURPUT FILE] -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
	run: [OURPUT FILE]
	files: [SOURCE FILES]
	log: [OURPUT FILE LOG]
	output: debug/


The ``compile`` indicates the command to compile the code, and ``run`` indicates the command to run the code. The ``files`` indicates the files to be examined by Promise (by default, all the .cc files). The ``log`` indicates the log file (no log file if this is not defined). The `output` indicates where the transformed code. 


Mark the code
--------------

Use ``__PROMISE__`` to mark the variables to be used in low precision, and use ``PROMISE_CHECK_VAR([VARIABLE])`` and ``PROMISE_CHECK_ARRAY([ARRAY},[NUM OF ELEMENTS])`` to mark to variabke or array to be checked; 



Additionally, one can define their customized low precisions in ``fp.json``. The built-in precisions are: 'h', 's', 'd' (half, single, double precision). Users can also define their own precisions (e.g., bfloat16) by adding new letters in ``fp.json``. 
A sample file is shown below:

.. parsed-literal::

	{   
	"c": [5, 2],
	"b": [8, 7],
	"h": [5, 10],
	"s": [8, 23],
	"d": [11, 52]
	}

Like above, "c" and "b" corresponding to E5M2 and bfloat16 precisions. 


Run the program in terminal
--------------------------------

In terminal, simply enter the command bellow: 

.. parsed-literal::

	get help:     promise --help | promise
        get version:  promise --version
	run program:  promise --precs=(customized precisions/built precisions) [options]


Options:

.. parsed-literal::

	-h --help                     Show this screen.
	--version                     Show version.
	--precs=<strs>                Set the precision following the built-in or cutomized precision letters [default: sd]
	--conf CONF_FILE              Get the configuration file [default: promise.yml]
	--fp FPT_FILE                 Get the file for floating point number format [default: fp.json]
	--output OUTPUT               Set the path of the output (where the result files are put)
	--verbosity VERBOSITY         Set the verbosity (betwen 0  and 4 for very low level debug) [default: 1]
	--log LOGFILE                 Set the log file (no log file if this is not defined)
	--verbosityLog VERBOSITY      Set the verbosity of the log file
	--debug                       Put intermediate files into `debug/` (and `compileErrors/` for compilation errrors) and display the execution trace when an error comes
	--run RUN                     File to be run
	--compile COMMAND             Command to compile the code
	--files FILES                 List of files to be examined by Promise (by default, all the .cc files)
	--nbDigits DIGITS             General required number of digits
	--path PATH                   Set the path of the project (by default, the current path)
	--pause                       Do pause between steps
	--parsing                     Parse the C file (without this, __PROMISE__ are replaced and that's all)
	--auto                        enable auto-instrumentation of source code
	--relError THRES              use criteria of precision relative error less than THRES instead of number of digits
	--noCadna                     will not use cadna, reference result computed in (non-stochastic) double precision
	--alias ALIAS                 Allow aliases (examples "g++=g++-14") [default:""]
	--CC        				          Set compiler for C program [default: g++]
	--CXX                         Set compiler for C++ program [default: g++]
	--plot                        Enable plotting of results [default: 1]


For detailed examples, see the `Examples <./EXAMPLE.rst>`_ document.

-------------------
Acknowledgements
-------------------

``cadnaPromise`` is based on `Promise2 <https://gitlab.lip6.fr/hilaire/promise2>`_  (Hilaire et al), a full rewriting of the first PROMISE version (Picot et al).

This work was supported by the France 2030 NumPEx Exa-MA (ANR-22-EXNU-0002) project managed by the French National Research Agency (ANR).
``Promise2`` has been developed with the financial support of the COMET project Model-Based Condition Monitoring and Process Control Systems, hosted by the Materials Center Leoben Forschung GmbH.
