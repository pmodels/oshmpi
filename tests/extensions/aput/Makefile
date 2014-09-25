CC       = cc
CFLAGS   = -g -Wall -O2 -std=c99
LD       = $(CC)
LDFLAGS  = $(CFLAGS)
LIBS     = -lhugetlbfs

all: armci-strided.x dmapp-strided.x

%.x: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f *.x

