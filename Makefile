LIBRARY  = libshmem.a
HEADERS  = shmem.h shmem-internals.h shmem-wait.h mcs-lock.h
SOURCES  = shmem.c shmem-internals.c bmalloc.c mcs-lock.c 
OBJECTS  = $(SOURCES:.c=.o)
TESTS    = test_start.c test_etext.c test_sheap.c
TESTS   += lat_bw.c barrier_performance.c put_performance.c get_performance.c osu_oshm_put_mr.c
BINARIES = $(TESTS:.c=.x)
MACCRAP  = $(BINARIES:.x=.x.dSYM)

CC      = mpicc
CFLAGS  = -g -O3 -std=c99 -Wall -I. #-DSHMEM_DEBUG=3 #-DSHEAP_HACK=2 #-DSHMEM_DEBUG=9
#CFLAGS  = -g -O0 -std=c99 -Wall -I. -DSHMEM_DEBUG=9
LDFLAGS = $(CFLAGS) $(LIBRARY)

all: $(OBJECTS) $(LIBRARY) $(BINARIES)

$(LIBRARY): $(OBJECTS) 
	$(AR) $(ARFLAGS) $(LIBRARY) $(OBJECTS)

# Makefile dependency ensures code is recompiled when flags change
%.x: %.c $(LIBRARY) $(HEADERS) Makefile
	$(CC) $(CFLAGS)    $< $(LDFLAGS) -o $@

# Makefile dependency ensures code is recompiled when flags change
%.o: %.c $(HEADERS) Makefile
	$(CC) $(CFLAGS) -c $<            -o $@

clean:
	-rm -f  $(OBJECTS)
	-rm -fr $(MACCRAP)

realclean: clean
	-rm -f  $(LIBRARY)
	-rm -f  $(BINARIES)
