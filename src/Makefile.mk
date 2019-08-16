#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS += -I$(top_builddir)/include -I$(top_srcdir)/include

include $(top_srcdir)/src/include/Makefile.mk
include $(top_srcdir)/src/internal/Makefile.mk
include $(top_srcdir)/src/shmem/Makefile.mk
include $(top_srcdir)/src/shmemx/Makefile.mk
