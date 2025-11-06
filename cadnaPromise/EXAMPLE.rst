Example
--------------------------------

We take an toy example here. Assuming original code is as below:

.. code-block:: cpp 

	#include <iostream>

	double sumVariables(double a, double b) {
		return a + b;
	}

	void sumArrays(double* arr1, double* arr2, double* result, int size) {
		for (int i = 0; i < size; i++) {
			result[i] = arr1[i] + arr2[i];
		}
	}

	int main() {
		int size = 5;
		double* arr1 = new double[size] {1.112, 2.2392, 3.315, 4.436, 5.5};
		double* arr2 = new double[size] {6.63, 7.717, 8.82, 9.9, 10.141};
		double* result = new double[size];

		double var1 = 15.5, var2 = 26.7212;
		

		sumArrays(arr1, arr2, result, size);
		
		double result_var = sumVariables(var1, var2);
		std::cout << "Sum of variables: " << result_var << std::endl;

		std::cout << "Sum of arrays: ";
		for (int i = 0; i < size; i++) {
			std::cout << result[i] << " ";
		}
		std::cout << std::endl;

		delete[] arr1;
		delete[] arr2;
		delete[] result;

		return 0;
	}


A sample configuration file is shown below:


.. parsed-literal::

	compile:
	- g++ toy.cpp -frounding-math -m64 -o toy.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include

	run: toy.out
	files: toy.cpp
	log: toy.log
	output: debug/


The marked code is as below, where ``__PROMISE__`` is used to mark the variables to be used in low precision, and ``PROMISE_CHECK_VAR([VARIABLE])`` and ``PROMISE_CHECK_ARRAY([ARRAY},[NUM OF ELEMENTS])`` are used to mark to variabke or array to be checked; here we use three different precisions for demonstration purpose (``__PR_1__``, ``__PR_2__``, ``__PR_3__``):


.. code-block:: cpp 

	#include <iostream>

	__PROMISE__ sumVariables(__PROMISE__ a, __PROMISE__ b) {
		return a + b;
	}

	void sumArrays(__PROMISE__* arr1, __PROMISE__* arr2, __PROMISE__* result, int size) {
		for (int i = 0; i < size; i++) {
			result[i] = arr1[i] + arr2[i];
		}
	}

	int main() {
		int size = 5;
		__PR_1__* arr1 = new __PR_1__[size] {1.112, 2.2392, 3.315, 4.436, 5.5};
		__PR_2__* arr2 = new __PR_2__[size] {6.63, 7.717, 8.82, 9.9, 10.141};
		__PR_3__* result = new __PR_3__[size];

		__PROMISE__ var1 = 15.5, var2 = 26.7212;
		__PROMISE__ result_var = sumVariables(var1, var2);

		sumArrays(arr1, arr2, result, size);
		std::cout << "Sum of variables: " << result_var << std::endl;
		
		// PROMISE_CHECK_VAR(result_var);
		PROMISE_CHECK_ARRAY(result, size);
		std::cout << "Sum of arrays: ";
		for (int i = 0; i < size; i++) {
			std::cout << result[i] << " ";
		}
		std::cout << std::endl;

		delete[] arr1;
		delete[] arr2;
		delete[] result;

		return 0;
	}


Assuming the file name is ``toy.cpp``, the configuration file ``promise.yml`` is as below:

.. parsed-literal::

	compile:
	- g++ toy.cpp -O3 -frounding-math -m64 -o toy.out -lcadnaC -L$CADNA_PATH/lib -I$CADNA_PATH/include

	run: toy.out
	files: toy.cpp
	log: toy.log
	output: debug/



Then one can run the program with the command below in terminal:

.. parsed-literal::

  promise --precs=cbhsd --conf=promise.yml --log=toy.log --output=debug/ --nbDigits=5


The output is as below:
.. parsed-literal::

	check compilers: g++
	🤘 cadnaPromise 🤘
	Using the compiler: g++
	We are working with 1 file and 11 different types
	The expectation is 5 digits.
	a) Get a reference result with cadna (double)
	b) Check with highest format (Double)
	c) Delta-Debug Single/Double
	The format float is enough for your expectation
																					d) Delta-Debug Half/Single
	The final result contains 11x float.                                            
	It tooks 23.71s                                                                 
	👉 29 compilations (26 failed) for 21.38s
	👉 3 executions   (0 failed) for 0.02s
	{'float': {0, 1, 2, 3, 4, 5, 6, '3', 7, '2', '1'}}


And the transformed code is located in ``output`` folder (here is ``debug/toy.cpp``):

One can also run the PROMISE with varying specified digits and generate pretty plots. For example, to run the program with precisions from 1 to 3 signifiant digits, and generate plots, one can use the command below:

.. parsed-literal::

  promise-batch --precs=cbhsd --nbDigits=1:3


If one want the generated plot, simply add ``--plot`` in the command line. 

