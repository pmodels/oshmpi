TESTS    = accessible_ping.c adjacent_32bit_amo.c atomic_inc.c barrier.c barrier_performance.c bcast.c bcast_flood.c big_reduction.c bigget.c bigput.c broadcast32_performance.c circular_shift.c collect32_performance.c cpi.c cswap.c fcollect32_performance.c fcollect64.c get1.c get_g.c get_performance.c hello.c ipgm.c iput-iget.c iput128.c iput32.c iput64.c iput_double.c iput_float.c iput_long.c iput_longdouble.c iput_longlong.c iput_short.c lfinc.c max_reduction.c msgrate.c netpipe.c ns.c ping.c pingpong-short.c pingpong.c ptp.c put1.c put_performance.c set_lock.c shmalloc.c shmem.c shmem_2dheat.c shmem_ct.c shmem_daxpy.c shmem_heat_image.c shmem_matrix.c shmemalign.c shmemlatency.c shrealloc.c spam.c sping.c strided_put.c swap1.c swapm.c test_lock.c test_shmem_accessible.c test_shmem_atomics.c test_shmem_barrier.c test_shmem_broadcast.c test_shmem_collects.c test_shmem_get.c test_shmem_get_globals.c test_shmem_get_shmalloc.c test_shmem_lock.c test_shmem_put.c test_shmem_put_globals.c test_shmem_put_shmalloc.c test_shmem_reduction.c test_shmem_synchronization.c test_shmem_zero_get.c test_shmem_zero_put.c to_all.c waituntil.c 
OBJECTS  = $(TESTS:.c=.o)
BINARIES = $(TESTS:.c=.x)
MACCRAP  = $(BINARIES:.x=.x.dSYM)

CC      = mpicc
CFLAGS  = -g -O3 -std=c99 -Wall -I..
LDFLAGS = $(CFLAGS) -L.. -lshmem

all: $(BINARIES)

%.x: %.o ../libshmem.a 
	$(CC) $(CFLAGS)    $< $(LDFLAGS) -o $@

%.o: %.c ../shmem.h Makefile
	$(CC) $(CFLAGS) -c $<            -o $@

clean:
	-rm -f  $(OBJECTS)
	-rm -fr $(MACCRAP)

realclean: clean
	-rm -f  $(BINARIES)

