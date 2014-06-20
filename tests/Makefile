SRCS := $(wildcard *.c)
PRGS := $(patsubst %.c,%,$(SRCS))
PRG_SUFFIX=.exe
BINS := $(patsubst %,%$(PRG_SUFFIX),$(PRGS))
OBJS := $(patsubst %,%.o,$(PRGS))

#PUT YOUR OSHMPI INSTALL PATH HERE!
CFLAGS		= -O2 -Wall -std=c99 -D_GNU_SOURCE -I/home/sg/builds/oshmpi/include
LDFLAGS 	= -L/home/sg/builds/oshmpi/lib -lshmem
OSHCC 		=  mpicc
OSHCXX 		=  mpicxx
RUNCMD 		=  mpiexec
NPROCS		= 2
CC 		= $(OSHCC)

.PHONY: all clean run

all : $(BINS)

OBJ = $(patsubst %$(PRG_SUFFIX),%.o,$@)
BIN = $@

%$(PRG_SUFFIX) : $(OBJS)
	$(OSHCC) $(CFLAGS) $(OBJ) -o $(BIN) $(LDFLAGS)

run :
	for bin in $(BINS); do \
		$(RUNCMD) -np $(NPROCS) ./$$bin; \
	done

clean:
	$(RM) $(BINS) $(OBJS)
