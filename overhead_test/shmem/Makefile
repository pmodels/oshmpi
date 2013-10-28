CFLAGS		= -O3 -g -Wall -std=gnu99 
LINKS 		= -Wl,-rpath=/soft/libraries/mpi/mvapich/3.0.a/gcc/lib -L/soft/libraries/mpi/mvapich/3.0.a/gcc/lib -lmpich
SHMLIB 		= -L/home/sghosh/projects/oshmpi -lshmem
CC 		= /soft/libraries/mpi/mvapich/3.0.a/gcc/bin/mpicc
RUNCMD 		= /soft/libraries/mpi/mvapich/3.0.a/gcc/bin/mpiexec
RUNOPT		=
NCOREOPT	= 
NCORE		=
NPROCOPT	= -np
NPROC		= 2
HEAP_GLOBAL	= heap
INCLUDES 	= -I. -I/home/sghosh/projects/oshmpi 

C_TESTS		=  osu_oshm_put_mr.c
C_EXES= $(C_TESTS:.c=.x)
EXES= $(C_EXES)

.PHONY: clean all common run

all: common $(EXES)

.SUFFIXES: .x

common:
	$(CC) $(INCLUDES) $(CFLAGS) -c osu_common.c

%.x: %.c
	$(CC) $(INCLUDES) $(CFLAGS) -o $@ $^ osu_common.o $(LINKS) $(SHMLIB)

run: 	all $(EXES)
	for exe in $(EXES); do \
		$(RUNCMD) $(RUNOPT) $(NPROCOPT) $(NPROC) ./$$exe $(HEAP_GLOBAL); \
	done

clean:
	$(RM) $(EXES) tmp.image *.o
