#include <iostream>
#include <cadna.h>
#include <omp.h>

int main(int argc, char **argv){

    std::cout << "-----------------------\n";
    std::cout << "print parameters\n";
	
    int i;
	omp_set_num_threads(4);         //Set Thread Number
#pragma omp parallel for        
	for (i = 0; i < 8; i++)
	{
		printf("%d Hello, World! Thread ID: %d\n", i,
			omp_get_thread_num());  //thread id
	}

    for(i=0; i<argc; i++)
    {
       printf("%s  ", argv[i]);
    }

    float h1 = atof(argv[1]);
    float h2 = atof(argv[2]);

    float h3 = h1 - h2;

    printf("\nThe sum of the two values is: %.10f\n", h3);
    std::cout << "\n-----------------------\n";
    return 0;
}