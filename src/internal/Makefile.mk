#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS += -I$(top_srcdir)/src/internal

noinst_HEADERS += src/internal/mem_impl.h        \
                  src/internal/coll_impl.h       \
                  src/internal/rma_impl.h

liboshmpi_la_SOURCES += src/internal/setup_impl.c

include $(top_srcdir)/src/internal/util/Makefile.mk