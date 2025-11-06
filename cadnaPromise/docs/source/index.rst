
***************************
PROMISE: PRecision OptiMISE
***************************



.. image:: _static/promise.png
  :width: 400
  :alt: PROMISE v2
  :align: center

Most numerical simulations are performed in double precision (IEEE754 binary64). Unfortunately this can be costly in terms of computing time, memory transfer and energy consumption.

PROMISE is a tool to auto-tune the precision of floating-point variables in numerical codes [GJPF2019]_. From an initial C or C++ program and a required accuracy on the computed results, PROMISE automatically modifies the floating-point types in favor of low precision variables. To estimate the numerical quality of results, PROMISE uses `CADNA <http://cadna.lip6.fr/>`__ which controls round-off errors in simulation programs [JC2008]_  [EBFJ2015]_. The search for a valid type configuration is based on the Delta Debug algorithm [Z2009]_.

PROMISE has been successfully tested on programs implementing several numerical algorithms including linear system solving and also on an industrial code that solves the neutron transport equations.

.. toctree::
    :hidden:

    self
    install
    usage
    examples


.. only:: html

    Usage
    =====

    With a few manipulations, you can transform your C/C++ code into a code that PROMISE can handle (you just need to tell PROMISE which variables it can change). Then PROMISE uses CADNA to evaluate the numerical quality of the results, and find (if it exists) a set of datatypes that satisfies your numerical expectactions.

        a) first :ref:`install<install>` PROMISE ;
        b) :ref:`Read<usage>` the details of the options ;
        c) And :ref:`check the example<examples>`.


    Contact
    =======

    If you have any feedback, bug reports or comments, please report it in our `gitlab <https://gitlab.lip6.fr/hilaire/promise2>`__ or send an email to the `CADNA team <mailto:cadna-team[at]lip6.fr>`__.


Acknowledgements
===================

Baptiste Mary has done the PROMISE logo (and also the CADNA logo).

This PROMISE version has been developed with the financial support of the COMET project *Model-Based Condition Monitoring and Process Control Systems*, hosted by the Materials Center Leoben Forschung GmbH.

This PROMISE version is a full rewriting of the first PROMISE version, written by Romain Picot *et al.*

.. only:: html
	  
   |

.. [GJPF2019]  S. Graillat, F. Jézéquel, R. Picot, F. Févotte, B. Lathuilière. Auto-tuning for floating-point precision with Discrete Stochastic Arithmetic, Journal of Computational Science, 36, pages 101017, 2019. HAL ID:hal-01331917 (`pdf <https://hal.archives-ouvertes.fr/hal-01331917>`__)
.. [JC2008]  F. Jézéquel and J.-M. Chesneaux. CADNA: a library for estimating round-off error propagation. Computer Physics Communications, 178(12):933-955, 2008.
.. [EBFJ2015] P. Eberhart, J. Brajard, P. Fortin, and F. Jézéquel, High Performance Numerical Validation using Stochastic Arithmetic, Reliable Computing, 21, pages 35-52, 2015 (`pdf <https://interval.louisiana.edu/reliable-computing-journal/volume-21/reliable-computing-21-pp-035-052.pdf>`__)
.. [Z2009]  A. Zeller, Why Programs Fail, 2nd ed., Morgan Kaufmann, Boston, 2009.

