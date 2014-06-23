#
# Copyright (C) 2014. See LICENSE in top-level directory.
#

check_PROGRAMS += tests/barrier_performance \
                  tests/get_performance \
                  tests/lat_bw \
                  tests/osu_oshm_put_mr \
                  tests/put_performance \
                  tests/test_etext \
                  tests/test_sheap \
                  tests/test_start \
                  tests/test_atomics \
                  tests/test_swap_cswap \
                  # end

TESTS += tests/barrier_performance \
         tests/get_performance \
         tests/lat_bw \
         tests/osu_oshm_put_mr \
         tests/put_performance \
         tests/test_etext \
         tests/test_sheap \
         tests/test_atomics \
	 tests/test_swap_cswap \
         # end

tests_barrier_performance_LDADD = libshmem.la
tests_get_performance_LDADD = libshmem.la
tests_lat_bw_LDADD = libshmem.la
tests_osu_oshm_put_mr_LDADD = libshmem.la
tests_put_performance_LDADD = libshmem.la
tests_test_etext_LDADD = libshmem.la
tests_test_sheap_LDADD = libshmem.la
tests_test_start_LDADD = libshmem.la
tests_test_atomics_LDADD = libshmem.la
tests_test_swap_cswap_LDADD = libshmem.la
