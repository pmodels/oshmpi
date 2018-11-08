#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#
CC=@OSHCC@
noinst_PROGRAMS = \
    hello         \
    barrier       \
    threaded_init_multiple   \
    threaded_init_serialized \
    threaded_init_funneled   \
    threaded_init_single

threaded_init_multiple_SOURCES = threaded_init.c
threaded_init_multiple_CPPFLAGS  = $(AM_CPPFLAGS)

threaded_init_serialized_SOURCES = threaded_init.c
threaded_init_serialized_CPPFLAGS  = -DSHMEM_REQUIRE_THREAD_SERIALIZED $(AM_CPPFLAGS)

threaded_init_funneled_SOURCES = threaded_init.c
threaded_init_funneled_CPPFLAGS  = -DSHMEM_REQUIRE_THREAD_FUNNELED $(AM_CPPFLAGS)

threaded_init_single_SOURCES = threaded_init.c
threaded_init_single_CPPFLAGS  = -DSHMEM_REQUIRE_THREAD_SINGLE $(AM_CPPFLAGS)