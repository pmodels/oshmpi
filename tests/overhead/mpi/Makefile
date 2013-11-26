CFLAGS		= -O3 -g -Wall -std=gnu99 
LINKS 		= 
SHMLIB 		= #-L/home/sghosh/projects/oshmpi -lshmem
CC 		= mpicc
RUNCMD 		= mpiexec
RUNOPT		=
NCOREOPT	= 
NCORE		=
NPROCOPT	= -np
NPROC		= 2
INCLUDES 	= -I. -I../..

C_TESTS		=  osu_mpi_put_mr.c
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
		$(RUNCMD) $(RUNOPT) $(NPROCOPT) $(NPROC) ./$$exe; \
	done

clean:
	$(RM) $(EXES) tmp.image *.o
