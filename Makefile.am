#
# Copyright (C) 2010. See COPYRIGHT in top-level directory.
#

ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(top_srcdir)/src -DONLY_MSPACES=1

lib_LTLIBRARIES = libshmem.la

libshmem_la_SOURCES = src/shmem-internals.c      \
                      src/shmem.c                \
                      src/oshmpi-mcs-lock.c      \
                      src/dlmalloc.c             \
                      src/shmemx-counting-put.c  \
                      src/shmemx-armci-strided.c

#libshmem_la_LDFLAGS = -version-info $(libshmem_abi_version)

include_HEADERS = src/shmem.h \
		  src/shmemx.h \
		  src/shmem-internals.h \
		  src/shmem-wait.h \
		  src/dlmalloc.h \
		  src/compiler-utils.h \
		  src/type_contiguous_x.h \
		  src/oshmpi-mcs-lock.h

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

bin_SCRIPTS = oshcc
CLEANFILES  = oshcc
# This does not do what I want (copy to $BUILD_DIR/src)
EXTRA_DIST  = ${top_srcdir}/src/oshcompiler.in

do_subst = sed  -e 's|[@]OSHMPI_CC[@]|$(CC)|g' \
		-e 's|[@]OSHMPI_CXX[@]|$(CXX)|g' \
		-e 's|[@]OSHMPI_FC[@]|$(FC)|g' \
		-e 's|[@]OSHMPI_INCDIR[@]|$(includedir)|g' \
		-e 's|[@]OSHMPI_LIBDIR[@]|$(libdir)|g'

AM_V_SED = $(am__v_SED_@AM_V@)
am__v_SED_ = $(am__v_SED_@AM_DEFAULT_V@)
am__v_SED_0 = @echo "  SED     " $@;

oshcc: $(top_srcdir)/src/oshcompiler.in Makefile
	$(AM_V_SED)$(do_subst) -e 's|[@]LANG[@]|C|g' < $(top_srcdir)/src/oshcompiler.in > oshcc
	@chmod +x oshcc



