#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS += -I$(top_srcdir)/src/internal

noinst_HEADERS += src/internal/coll_impl.h       \
                  src/internal/coll_activeset_impl.h \
                  src/internal/rma_impl.h        \
                  src/internal/amo_impl.h        \
                  src/internal/am_impl.h         \
                  src/internal/am_pre.h          \
                  src/internal/amo_am_impl.h     \
                  src/internal/rma_am_impl.h     \
                  src/internal/am_progress_impl.h\
                  src/internal/order_impl.h      \
                  src/internal/p2p_impl.h        \
                  src/internal/strided_impl.h    \
                  src/internal/lock_impl.h

liboshmpi_la_SOURCES += src/internal/setup_impl.c  \
                        src/internal/mem_impl.c    \
                        src/internal/space_impl.c  \
                        src/internal/team_impl.c   \
                        src/internal/ctx_impl.c    \
                        src/internal/strided_impl.c\
                        src/internal/coll_impl.c   \
                        src/internal/am_impl.c     \
                        src/internal/amo_am_impl.c \
                        src/internal/rma_am_impl.c \
                        src/internal/common.c

include $(top_srcdir)/src/internal/util/Makefile.mk
