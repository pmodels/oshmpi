CC       = cc
CFLAGS   = -O2
LD       = $(CC)
LDFLAGS  = $(CFLAGS)
LIBS     = -lhugetlbfs

all: armci-strided.x

%.x: %.o
	$(LD) $(LDFLAGS) $< $(LIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm -f *.o

realclean: clean
	-rm -f *.x

distclean: realclean

