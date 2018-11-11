#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

ACLOCAL_AMFLAGS = -I confdb
AM_CPPFLAGS =

lib_LTLIBRARIES = liboshmpi.la
include_HEADERS = include/shmem.h       \
                  include/shmemx.h      \
                  include/shmemx.fh

external_subdirs = @opasrcdir@
external_ldflags = @opasrcdir@
convenience_libs = @opalib@

noinst_HEADERS =
bin_SCRIPTS =
CLEANFILES =
liboshmpi_la_SOURCES =
liboshmpi_la_LDFLAGS = $(external_ldflags)
liboshmpi_la_LIBADD = $(convenience_libs)
EXTRA_liboshmpi_la_DEPENDENCIES = $(convenience_libs)
EXTRA_DIST = 

include src/Makefile.mk
include env/Makefile.mk
include include/Makefile.mk

EXTRA_DIST += CHANGES COPYRIGHT maint autogen.sh

DIST_SUBDIRS = tests ${external_subdirs}
SUBDIRS = ${external_subdirs}
