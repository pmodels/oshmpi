#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS += -DONLY_MSPACES -I$(top_srcdir)/src/internal/util

noinst_HEADERS += src/internal/util/dlmalloc.h    \
                  src/internal/util/utlist.h      \
                  src/internal/util/thread.h      \
                  src/internal/util/gpu/cuda.h

liboshmpi_la_SOURCES += src/internal/util/dlmalloc.c \
                        src/internal/util/symm_mem.c \
                        src/internal/util/mem_pool.c \
                        src/internal/util/gpu/fallback.c
