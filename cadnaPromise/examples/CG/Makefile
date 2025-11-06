SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

LIB =$(CADNA_PATH)/lib
INCLUDE =$(CADNA_PATH)/include

FILE = cg.o


OBJS = print_results.o  randdp.o c_timers.o wtime.o

default: $(FILE) $(OBJS)
	g++ -O3 -mcmodel=medium -o cg ${FILE} ${OBJS} -lm -mcmodel=medium -lcadnaC -O3 -frounding-math -g -I $(INCLUDE) -L $(LIB)

.c.o:
	g++ -c $< -o $@ -lcadnaC -O3 -g -mcmodel=medium -frounding-math -I $(INCLUDE) -L $(LIB)


cg.o:		cg.c  globals.h npbparams.h

print_results.o: print_results.c

randdp.o : randdp.c

c_timers.o : c_timers.c

wtime.o: wtime.c

clean:
	rm -f *.o *~



