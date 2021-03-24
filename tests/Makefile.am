#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#
CC=@OSHCC@

check_PROGRAMS =  \
    hello         \
    barrier       \
    putmem_quiet              \
    putmem_quiet_stext        \
    iput_quiet                \
    iput                      \
    amo_set_fetch             \
    amo_set_wait_until        \
    int_put                   \
    int_put_heap              \
    int_amo                   \
    int_amo_heap              \
    threaded_init_multiple   \
    threaded_init_serialized \
    threaded_init_funneled   \
    threaded_init_single     \
    space                    \
    space_int_put            \
    space_ctx_int_put        \
    space_int_amo

noinst_HEADERS = gpu_common.h

if OSHMPI_ENABLE_CUDA_TEST
check_PROGRAMS += space_cuda               \
                  space_int_put_cuda       \
                  space_ctx_int_put_cuda   \
                  space_int_amo_cuda
endif
if OSHMPI_ENABLE_ZE_TEST
check_PROGRAMS += space_ze
endif

TESTS = $(check_PROGRAMS)

MPIEXEC=mpiexec -n 2
LOG_COMPILER=$(MPIEXEC)

int_put_heap_SOURCES = int_put.c
int_put_heap_CPPFLAGS  = -DUSE_SYMM_HEAP $(AM_CPPFLAGS)

int_amo_heap_SOURCES = int_amo.c
int_amo_heap_CPPFLAGS  = -DUSE_SYMM_HEAP $(AM_CPPFLAGS)

threaded_init_multiple_SOURCES = threaded_init.c
threaded_init_multiple_CPPFLAGS  = $(AM_CPPFLAGS)

threaded_init_serialized_SOURCES = threaded_init.c
threaded_init_serialized_CPPFLAGS  = -DSHMEM_REQUIRE_THREAD_SERIALIZED $(AM_CPPFLAGS)

threaded_init_funneled_SOURCES = threaded_init.c
threaded_init_funneled_CPPFLAGS  = -DSHMEM_REQUIRE_THREAD_FUNNELED $(AM_CPPFLAGS)

threaded_init_single_SOURCES = threaded_init.c
threaded_init_single_CPPFLAGS  = -DSHMEM_REQUIRE_THREAD_SINGLE $(AM_CPPFLAGS)

putmem_quiet_stext_SOURCES = putmem_quiet.c
putmem_quiet_stext_CPPFLAGS  = -DTEST_SYMM_TEXT $(AM_CPPFLAGS)

if OSHMPI_ENABLE_CUDA_TEST
space_cuda_SOURCES = space.c
space_cuda_CPPFLAGS  = -DUSE_CUDA $(AM_CPPFLAGS)

space_int_put_cuda_SOURCES = space_int_put.c
space_int_put_cuda_CPPFLAGS  = -DUSE_CUDA $(AM_CPPFLAGS)

space_ctx_int_put_cuda_SOURCES = space_ctx_int_put.c
space_ctx_int_put_cuda_CPPFLAGS  = -DUSE_CUDA $(AM_CPPFLAGS)

space_int_amo_cuda_SOURCES = space_int_amo.c
space_int_amo_cuda_CPPFLAGS  = -DUSE_CUDA $(AM_CPPFLAGS)
endif

if OSHMPI_ENABLE_ZE_TEST
space_ze_SOURCES = space.c
space_ze_CPPFLAGS  = -DUSE_ZE $(AM_CPPFLAGS)
endif

.PHONY: checkprogs
checkprogs: $(check_PROGRAMS)