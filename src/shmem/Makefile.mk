#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

liboshmpi_la_SOURCES += src/shmem/setup.c    \
                        src/shmem/mem.c      \
                        src/shmem/rma.c      \
                        src/shmem/rma_typed.c\
                        src/shmem/rma_sized.c\
                        src/shmem/amo_std_typed.c    \
                        src/shmem/amo_ext_typed.c    \
                        src/shmem/amo_bitws_typed.c  \
                        src/shmem/coll.c                 \
                        src/shmem/reduce_minmax_typed.c  \
                        src/shmem/reduce_sumprod_typed.c \
                        src/shmem/reduce_bitws_typed.c   \
                        src/shmem/p2p.c                  \
                        src/shmem/p2p_typed.c            \
                        src/shmem/order.c                \
                        src/shmem/lock.c
