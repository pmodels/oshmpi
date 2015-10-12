#
# Copyright (C) 2010. See COPYRIGHT in top-level directory.
#

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/src -DONLY_MSPACES=1

lib_LTLIBRARIES = libshmem.la

libshmem_la_SOURCES = src/shmem-internals.c      \
                      src/shmem.c                \
                      src/lock.c                 \
                      src/dlmalloc.c

#libshmem_la_LDFLAGS = -version-info $(libshmem_abi_version)

include_HEADERS = src/shmem.h src/shmem-internals.h src/dlmalloc.h

bin_PROGRAMS =
check_PROGRAMS = 
TESTS = 
XFAIL_TESTS = 

MPIEXEC = mpiexec -n 2
TESTS_ENVIRONMENT = $(MPIEXEC)

#include benchmarks/Makefile.mk
include tests/Makefile.mk

# not working
#include tests/thirdparty/Makefile.mk

.PHONY: checkprogs
checkprogs: $(check_PROGRAMS)
