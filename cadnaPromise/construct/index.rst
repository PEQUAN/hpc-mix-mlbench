Promise: PRecision optiMISE
===========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


About PROMISE
-------------
Most numerical simulations are performed in double precision (IEEE754 binary64). Unfortunately this can be costly in terms of computing time, memory transfer and energy consumption.

PROMISE is a tool to auto-tune the precision of floating-point variables in code [GJPF2016]_. From an initial C or C++ program and a required accuracy on the computed results, PROMISE automatically modifies the floating-point types and maximizes the number of single precision variables. To estimate the numerical quality of results, PROMISE uses `CADNA <http://cadna.lip6.fr/>`_ which controls round-off errors in simulation programs [JC2008]_.

PROMISE has been successfully tested on programs implementing several numerical algorithms including linear system solving and also on an industrial code that solves the neutron transport equations.


Installation
------------




Examples
--------
Promise code contains various examples. See ??


Contact
-------
If you have any feedback, bug reports or comments, please report it in our `gitlab <https://gitlab.lip6.fr/hilaire/promise2>`_ or send an email to the `CADNA team <mailto:cadna-team[at]lip6.fr>`_.

.. [GJPF2016] S. Graillat, F. Jézéquel, R. Picot, F. Févotte, B. Lathuilière. Auto-tuning for floating-point precision with Discrete Stochastic Arithmetic. 2016. HAL ID:hal-01331917 (`pdf <https://hal.archives-ouvertes.fr/hal-01331917>`_)
.. [JC2008] F. Jézéquel and J.-M. Chesneaux. CADNA: a library for estimating round-off error propagation. Computer Physics Communications, 178(12):933-955, 2008.


.. * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
