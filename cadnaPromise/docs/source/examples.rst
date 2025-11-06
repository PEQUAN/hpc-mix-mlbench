
.. _examples:

*******
Example
*******

We use the same program for arclength computation as in [RNND2013]_ with all floating-point variables initially declared in double precision. 


**Initial code:**

.. code-block:: c++

    double fun(double x){
      int k, n = 5;
      double t1;
      double d1 = 1.0;

      t1 = x;
      for ( k = 1; k <= n; k++ )
        {
          d1 = 2.0 * d1;
          t1 = t1+ sin(d1 * x)/d1;
        }
      return t1;
    }

    int main( int argc, char **argv) {

      int i,n = 1000000;
      double h;
      double t1, t2, dppi;
      double s1;
      std::ofstream res;
      std::cout.precision(15);

      t1 = -1.0;
      dppi = acos(t1);
      s1 = 0.0;
      t1 = 0.0;
      h = dppi / n;

      for ( i = 1; i <= n; i++)
        {
          t2 = fun(i * h);
          s1 = s1 + sqrt(h*h + (t2 - t1) * (t2 - t1));
          t1 = t2;
          //if (i%1000==0) PROMISE_CHECK_VAR(t1);
        }

      std::cout << s1 << std::endl;

      return 0;
    }

Result: 5.79577632241303

Result printed using CADNA: 0.579577632241E+001 (other digits are affected by round-off errors)	
    
**Code modified to be used with PROMISE:**

To run an example, first you should replace every variable type you want to test by ``__PROMISE__``. Also, if you want to link the types of some variables (for example link the type of arguments between the function calls and their prototypes) those types should be replaced by ``__PR_XXXX__`` (``XXXX`` is user defined).
In a second step, you should specify a variable whose accuracy should be checked by PROMISE. For this purpose, if the variable(``var``) is an array you should insert this kind of instruction after the computation of ``var``:

.. code-block:: c

    PROMISE_CHECK_ARRAY(var, size of var);

Otherwise, you need to insert the following instruction:

.. code-block:: c

    PROMISE_CHECK_VAR(var);

.. code-block:: c
    :emphasize-lines: 1,3,4,18,19,20,39

    __PR_fun__ fun(__PR_1__ x){
      int k, n = 5;
      __PR_fun__ t1;
      __PR_1__ d1 = 1.0;

      t1 = x;
      for ( k = 1; k <= n; k++ )
        {
          d1 = 2.0 * d1;
          t1 = t1+ sin(d1 * x)/d1;
        }
      return t1;
    }

    int main( int argc, char **argv) {

      int i,n = 1000000;
      __PR_1__ h;
      __PROMISE__ t1, t2, dppi;
      __PROMISE__ s1;
      std::ofstream res;
      std::cout.precision(15);

      t1 = -1.0;
      dppi = acos(t1);
      s1 = 0.0;
      t1 = 0.0;
      h = dppi / n;

      for ( i = 1; i <= n; i++)
        {
          t2 = fun(i * h);
          s1 = s1 + sqrt(h*h + (t2 - t1) * (t2 - t1));
          t1 = t2;
          //if (i%1000==0) PROMISE_CHECK_VAR(t1);
        }

      std::cout << s1 << std::endl;
      PROMISE_CHECK_VAR(s1);
      return 0;
    }

Now, you can run PROMISE for Half/Single/Double mixed-precision with this command::

    runPromise hsd

You can modify the number of requested digits (``nbDigits``) in the ``promise.yml`` file.

**Code after the use of PROMISE (6 requested digits):**

The modified source code can be found in the directory which has been specified in the ``output`` option in the ``promise.yml`` file. If this option is not included in the ``promise.yml``, by default, a directory named ``result`` will contain the modified source code.

.. code-block:: c
    :emphasize-lines: 1,3,4,18,19,20

    double fun(double x){
      int k, n = 5;
      double t1;
      half_float::half d1; d1= 1.0;

      t1 = x;
      for ( k = 1; k <= n; k++ )
        {
          d1 = 2.0 * d1;
          t1 = t1+ sin(d1 * x)/d1;
        }
      return t1;
    }

    int main( int argc, char **argv) {

      int i,n = 1000000;
      double h;
      double t1;double t2;float dppi;
      double s1;
      std::ofstream res;
      std::cout.precision(15);

      t1 = -1.0;
      dppi = acos(t1);
      s1 = 0.0;
      t1 = 0.0;
      h = dppi / n;

      for ( i = 1; i <= n; i++)
        {
          t2 = fun(i * h);
          s1 = s1 + sqrt(h*h + (t2 - t1) * (t2 - t1));
          t1 = t2;
          //if (i%1000==0) PROMISE_CHECK_VAR(t1);
        }

      std::cout << s1 << std::endl;

      return 0;
    }

Result: 5.79577686259398 (6 correct digits, *i.e.* in common with the CADNA result)


.. [RNND2013] C. Rubio-González, C. Nguyen, H.D. Nguyen, J. Demmel, W. Kahan, K. Sen, D.H. Bailey, C. Iancu, D. Hough, Precimonious: Tuning assistant for floating-point precision, in: Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis, SC '13, ACM, New York, NY, USA, 2013, pp. 27:1-27:12.
