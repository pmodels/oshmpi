#
# Copyright (C) 2010. See COPYRIGHT in top-level directory.
#

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/src

lib_LTLIBRARIES = libshmem.la

libshmem_la_SOURCES = src/shmem-internals.c      \
                      src/shmem.c                \
                      src/mcs-lock.c             \
                      src/bmalloc.c

#libshmem_la_LDFLAGS = -version-info $(libshmem_abi_version)

include_HEADERS = src/shmem.h src/shmem-internals.h

bin_PROGRAMS =
check_PROGRAMS = 
TESTS = 
XFAIL_TESTS = 

MPIEXEC = mpiexec -n 2
TESTS_ENVIRONMENT = $(MPIEXEC)

#include benchmarks/Makefile.mk
include tests/Makefile.mk

.PHONY: checkprogs
checkprogs: $(check_PROGRAMS)
