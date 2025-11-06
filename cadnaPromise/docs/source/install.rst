
.. highlight:: console

.. _install:

************
Installation
************

Requirements
============

PROMISE is a Python program with the following requisites:

* CADNA
* g++ compiler
* Python 3.5 or upper

It also requires the following packages ``setuptools``,  ``colorlog``, ``colorama``, ``docopt``, ``pyyaml`` and ``tqdm`` (``pytest`` is used for the unit tests and ``sphinx``,  and ``sphinx_bootstrap_theme`` for the documentation that will be installed automatically with PROMISE)


Get PROMISE and install it
==========================
.. You can install the latest release using pip::

    $ pip install cadnaPromise

You can directly download `the latest version <https://gitlab.lip6.fr/hilaire/promise2>`_ from our gitlab.
In that case, you can git clone the project, and run the installation::

    $ git clone https://gitlab.lip6.fr/hilaire/promise2.git
    $ cd promise2
    $ python setup.py install

and then a command named ``runPromise`` is installed (usually in ``/usr/local/bin``).


Run the tests
=============
In the PROMISE directory, you can run the tests with::

    $ cd examples
    $ pytest

All the tests should pass.

The tests rely on the  ``CADNA_PATH`` environment variable that should be set to the CADNA path (it depends on your installation).

Generate the documentation
==========================
The documentation (this website) can be generated with `sphinx <https://www.sphinx-doc.org>`_, from the PROMISE directory::

    $ cd doc
    $ make html

And the pdf documentation with::

    $ make latexpdf

