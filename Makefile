LIBRARY = libshmem.a
HEADERS = shmem.h 
SOURCES = shmem.c 
OBJECTS = $(SOURCES:.c=.o)
TESTS   =

CC      = mpicc
CFLAGS  = -g -O2 -std=c99 -Wall -I.
LDFLAGS = $(CFLAGS) $(LIBRARY)

all: $(LIBRARY) $(TESTS)

$(LIBRARY): $(OBJECTS) 
	$(AR) $(ARFLAGS) $(LIBRARY) $(OBJECTS)

%.x: %.c $(LIBRARY) shmem.h
	$(CC) $(CFLAGS)    $< $(LDFLAGS) -o $@

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $<            -o $@

clean:
	-rm -f $(OBJECTS)

realclean: clean
	-rm -f $(LIBRARY)

