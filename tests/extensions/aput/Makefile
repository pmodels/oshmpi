CC       = cc
CFLAGS   = -g -Wall -O2 -std=c99
LD       = $(CC)
LDFLAGS  = $(CFLAGS)
LIBS     = -lhugetlbfs

all: armci-strided.x

%.x: %.c
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

#%.x: %.o
#	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

#%.o: %.c
#	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x

distclean: realclean

