
.. _usage:

*****
Usage
*****

Modifying the code
==================

This section details how to use PROMISE. First you need to modify your source code to tell PROMISE which variables you want to be considered, and which output(s) should be considered for the final accuracy. And then, you can run PROMISE and get some results.


Defining the types to be considered
-----------------------------------
To use PROMISE on your C/C++ code, you first need to instrument your code and change the type of the variables you want to change in quad/double/single/half. This is done by introducing a ``__PROMISE__`` type, and you need to change manually [#manual]_ the type of these variables.
So

.. code-block:: c

    double a,b;

becomes

.. code-block:: c

    __PROMISE__ a,b;

All the variables defined with the ``__PROMISE__`` type will be examined, and different types (quad, double, float, half) will be tried. Note that ``a`` and ``b`` are defined with the ``__PROMISE__`` type but their types are not linked, and will not be necessarily the same.
In order to link types, another dedicated PROMISE type can be used, in the form ``__PR_xxxx__`` where ``xxxx`` is user defined. For example, to define that a variable must have the same type as the return type of a function, we can write

.. code-block:: c

    __PR_ret__ foo(int bar, double x) {
        __PR_ret__ y;
        y = x*bar;
        return y;
    }

If we use ``__PROMISE__`` for the return type of ``foo`` and for ``y``, PROMISE may try two different types. So, in order to have two variables ``a`` and ``b`` with the same type chosen by PROMISE, we must define

.. code-block:: c

    __PR_ab__ a,b;


Specifying the output(s)
------------------------
Two functions [#macro]_ are used to tell PROMISE which variables to check:

* ``PROMISE_CHECK_VAR`` to check a variable
* ``PROMISE_CHECK_ARRAY`` to check an array

So, if you want PROMISE to check the accuracy of the variable ``res`` at the end of a given computation, you need to add

.. code-block:: c

    PROMISE_CHECK_VAR(res);

after the computation.

If ``res`` is an array, then

.. code-block:: c

    PROMISE_CHECK_ARRAY(res, size);

should be used (where ``size`` is an integer with the array size). Then all the values of the array are checked to determine if they satisfy the accuracy requirement.

``PROMISE_CHECK_VAR`` can be called several times (with the same variable or not), and in that case, all the occurrences are checked against the requirements.


Run PROMISE
===========

The ``Promise.yml`` file
------------------------
Once your code is adapted to PROMISE, it's time to run PROMISE. For that, there is a script called ``runPromise`` that has been created (usually in ``/usr/local/bin/``) when installing PROMISE.
To run, PROMISE needs to know:

* where is your source code
* how to compile it
* how many digits you need for the specified output(s)


All these informations (and other options) are given in a specific ``yml`` config file, named ``promise.yml`` by default.
It should contains the following informations:

.. csv-table::
    :widths: 15, 100

    ``compile``, "a yml list of commands to compile the source code. The compilation should include the CADNA library, so options such as ``-lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include`` should be added to your ``g++`` command line. As a remark, CADNA requires  ``g++``, not ``gcc``."
    ``run``, executable name produced by the compilation process
    ``files``, comma-separated list of source files (files considered and parsed by PROMISE)
    ``nbDigits``, required (general) number of correct digits

The following informations are optional:

.. csv-table::
    :widths: 35, 100

    ``nbDigitsPerVariable``, "dedicated number of digits per variable (should be a yml dictionary: for each variable name, an integer is given). For the variables not defined here, the (general) number of digits given in ``nbDigits`` is used"
    ``log``, name of the log file (no log file if this is not defined)
    ``verbosity``, "verbosity level (integer between 0 and 4). 0 means minimum messages, 1 (default level) for some info messages, 2 for debug messages, 3 to display the command lines used and 4 to get the promise-dedicated outputs of the runs"
    ``verbosityLog``, verbosity level for the log file (the integer values already mentioned)
    ``output``, name of the output repository (``result/`` by default)

A minimum ``promise.yml`` file could be (for a simple project based on ``foo.cc`` and ``bar.cc`` and 8 expected digits):

.. code-block:: yaml

    compile:
    - g++ -c foo.cc -I$CADNA_PATH/include
    - g++ -c bar.cc -I$CADNA_PATH/include
    - g++ foo.o bar.o -o mytest.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
    run: mytest.out
    files: foo.cc, bar.cc
    nbDigits: 8

Using all the options, we can have something like (we expect 4 digits for the variable ``x``, 2 for ``foo`` and 8 for the other variables):

.. code-block:: yaml

    compile:
    - g++ -c foo.cc -I$CADNA_PATH/include
    - g++ -c bar.cc -I$CADNA_PATH/include
    - g++ foo.o bar.o -o mytest.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include
    run: mytest.out
    files: foo.cc, bar.cc
    nbDigits: 8
    nbDigitsPerVariable:
        x: 4
        foo: 2
    log: mylog.txt
    verbosity: 1
    verbosityLog: 4
    output: finalResult/


Command line usage
------------------


In addition to the parameters put in the ``promise.yml``, we need to tell PROMISE which floating-point tuning it should do.
The command ``runPromise`` is used to run PROMISE, followed with one of the three [#hsdq]_ possibilities:

* ``hsd`` for Half/Single/Double mixed-precision
* ``hs`` for Half/Single mixed-precision
* ``sd`` for Single/Double mixed-precision

So the command::

    $ runPromise sd

simply runs PROMISE for Single/Double mixed-precision using the parameters from the ``promise.yml`` file in the current directory.

The option ``--conf CONF_FILE`` can be used to indicate PROMISE that the file ``CONF_FILE`` should be used instead of ``promise.yml`` (used to specify the path of ``promise.yml`` for example).

The option ``--debug`` can also be used for developing PROMISE (it displays execution traces when an error arises and puts the intermediate results in the ``debug/`` folder.

The option ``--pause`` is used to force a pause between the different steps of the Delta Debug algorithm (sometimes used for debugging).

Moreover, all the previous options can *also* be specified in the command line, by prefixing them with ``--`` (if an option is specified in both, the command line value prevails). They are gathered in the usage documentation (obtained with ``runPromise --help``):

::

     Promise v2

    Usage:
        runPromise -h | --help
        runPromise (hsd|hs|sd) [options]

    Options:
      -h --help                     Show this screen.
      --conf CONF_FILE              get the configuration file [default: promise.yml]
      --output OUTPUT               set the path of the output (where the result files are put)
      --verbosity VERBOSITY         set the verbosity (betwen 0 display the minimum and 4 for very low level debug) [default: 1]
      --log LOGFILE                 set the log file (no log file if this is not defined)
      --verbosityLog VERBOSITY      set the verbosity of the log file
      --debug                       put intermediate files into `debug/` and display the execution trace when an error comes (only use during promise development)
      --run RUN                     file to be run
      --compile COMMAND             command to compile the code
      --files FILES                 list of files to be examined by PROMISE (by default, all the .cc files)
      --nbDigits DIGITS             general required number of correct digits
      --path PATH                   set the path of the project (by default, the current path)
      --pause                       do pause between steps
      hsd                           Half/Single/Double mixed-precision
      hs                            Half/Single mixed-precision
      sd                            Single/Double mixed-precision


Usually, the common options used for a project are put in the ``promise.yml`` file (like the compilation commands, the files, etc.), whereas the options frequently changed (for various attempts) are put in the command line (like the number of digits or the verbosity level). A typical ``runPromise`` command can be::

    runPromise hsd --nbDigits=5 --verbosity=2

.. warning::

    The parsing done to recognize the type of the variables (``__PROMISE__`` and ``__PR_xxx__``-like) is not robust to all the codes. For example, array declarations and initializations may not work.


.. [#manual] This can be easily automated if you want to change *all* the ``double`` variables in your code. This can also be done in some parts of the code. It's up to the user to do it for the moment.

.. [#macro] Technically, they are C macros, such that PROMISE can get the name of the variable and use it.

.. [#hsdq] Other options included *quad* floating-point arithmetic may be added (PROMISE intern code already supports it, but it has not been tested yet; contact us if you want this improvement)
