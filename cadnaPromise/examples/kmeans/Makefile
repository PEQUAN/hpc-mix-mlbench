# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O2 

kmeans: getopt.o kmeans.o evaluate.o 
	$(CC) $(CC_FLAGS) getopt.o kmeans.o evaluate.o  -o test

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c
	
evaluate.o: evaluate.c 
	$(CC) $(CC_FLAGS) evaluate.c -c

kmeans.o: kmeans.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans.c -c

clean:
	rm -f *.o *~ test
