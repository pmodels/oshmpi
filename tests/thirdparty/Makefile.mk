#
# Copyright (C) 2014. See LICENSE in top-level directory.
#

check_PROGRAMS += accessible_ping \
                  adjacent_32bit_amo \
                  atomic_inc \
                  barrier \
                  barrier_performance \
                  bcast \
                  bcast_flood \
                  big_reduction \
                  bigget \
                  bigput \
                  broadcast32_performance \
                  circular_shift \
                  collect32_performance \
                  cpi \
                  cswap \
                  fcollect32_performance \
                  fcollect64 \
                  get1 \
                  get_g \
                  get_performance \
                  hello \
                  ipgm \
                  iput-iget \
                  iput128 \
                  iput32 \
                  iput64 \
                  iput_double \
                  iput_float \
                  iput_long \
                  iput_longdouble \
                  iput_longlong \
                  iput_short \
                  lfinc \
                  max_reduction \
                  msgrate \
                  ns \
                  ping \
                  pingpong-short \
                  pingpong \
                  ptp \
                  put1 \
                  put_performance \
                  set_lock \
                  shmalloc \
                  shmem_2dheat \
                  shmem_daxpy \
                  shmem_heat_image \
                  shmem_matrix \
                  shmemalign \
                  shmemlatency \
                  shrealloc \
                  spam \
                  sping \
                  strided_put \
                  swap1 \
                  swapm \
                  test_lock \
                  test_shmem_accessible \
                  test_shmem_atomics \
                  test_shmem_barrier \
                  test_shmem_broadcast \
                  test_shmem_collects \
                  test_shmem_get \
                  test_shmem_get_globals \
                  test_shmem_get_shmalloc \
                  test_shmem_lock \
                  test_shmem_put \
                  test_shmem_put_globals \
                  test_shmem_put_shmalloc \
                  test_shmem_reduction \
                  test_shmem_synchronization \
                  test_shmem_zero_get \
                  test_shmem_zero_put \
                  to_all \
                  waituntil \
                  # end

TESTS += accessible_ping \
         adjacent_32bit_amo \
         atomic_inc \
         barrier \
         barrier_performance \
         bcast \
         bcast_flood \
         big_reduction \
         bigget \
         bigput \
         broadcast32_performance \
         circular_shift \
         collect32_performance \
         cpi \
         cswap \
         fcollect32_performance \
         fcollect64 \
         get1 \
         get_g \
         get_performance \
         hello \
         ipgm \
         iput-iget \
         iput128 \
         iput32 \
         iput64 \
         iput_double \
         iput_float \
         iput_long \
         iput_longdouble \
         iput_longlong \
         iput_short \
         lfinc \
         max_reduction \
         msgrate \
         ns \
         ping \
         pingpong-short \
         pingpong \
         ptp \
         put1 \
         put_performance \
         set_lock \
         shmalloc \
         shmem_2dheat \
         shmem_daxpy \
         shmem_heat_image \
         shmem_matrix \
         shmemalign \
         shmemlatency \
         shrealloc \
         spam \
         sping \
         strided_put \
         swap1 \
         swapm \
         test_lock \
         test_shmem_accessible \
         test_shmem_atomics \
         test_shmem_barrier \
         test_shmem_broadcast \
         test_shmem_collects \
         test_shmem_get \
         test_shmem_get_globals \
         test_shmem_get_shmalloc \
         test_shmem_lock \
         test_shmem_put \
         test_shmem_put_globals \
         test_shmem_put_shmalloc \
         test_shmem_reduction \
         test_shmem_synchronization \
         test_shmem_zero_get \
         test_shmem_zero_put \
         to_all \
         waituntil \
         # end

tests_thirdparty_accessible_ping_LDADD = libshmem.la
tests_thirdparty_adjacent_32bit_amo_LDADD = libshmem.la
tests_thirdparty_atomic_inc_LDADD = libshmem.la
tests_thirdparty_barrier_LDADD = libshmem.la
tests_thirdparty_barrier_performance_LDADD = libshmem.la
tests_thirdparty_bcast_LDADD = libshmem.la
tests_thirdparty_bcast_flood_LDADD = libshmem.la
tests_thirdparty_big_reduction_LDADD = libshmem.la
tests_thirdparty_bigget_LDADD = libshmem.la
tests_thirdparty_bigput_LDADD = libshmem.la
tests_thirdparty_broadcast32_performance_LDADD = libshmem.la
tests_thirdparty_circular_shift_LDADD = libshmem.la
tests_thirdparty_collect32_performance_LDADD = libshmem.la
tests_thirdparty_cpi_LDADD = libshmem.la
tests_thirdparty_cswap_LDADD = libshmem.la
tests_thirdparty_fcollect32_performance_LDADD = libshmem.la
tests_thirdparty_fcollect64_LDADD = libshmem.la
tests_thirdparty_get1_LDADD = libshmem.la
tests_thirdparty_get_g_LDADD = libshmem.la
tests_thirdparty_get_performance_LDADD = libshmem.la
tests_thirdparty_hello_LDADD = libshmem.la
tests_thirdparty_ipgm_LDADD = libshmem.la
tests_thirdparty_iput_iget_LDADD = libshmem.la
tests_thirdparty_iput128_LDADD = libshmem.la
tests_thirdparty_iput32_LDADD = libshmem.la
tests_thirdparty_iput64_LDADD = libshmem.la
tests_thirdparty_iput_double_LDADD = libshmem.la
tests_thirdparty_iput_float_LDADD = libshmem.la
tests_thirdparty_iput_long_LDADD = libshmem.la
tests_thirdparty_iput_longdouble_LDADD = libshmem.la
tests_thirdparty_iput_longlong_LDADD = libshmem.la
tests_thirdparty_iput_short_LDADD = libshmem.la
tests_thirdparty_lfinc_LDADD = libshmem.la
tests_thirdparty_max_reduction_LDADD = libshmem.la
tests_thirdparty_msgrate_LDADD = libshmem.la
tests_thirdparty_ns_LDADD = libshmem.la
tests_thirdparty_ping_LDADD = libshmem.la
tests_thirdparty_pingpong_short_LDADD = libshmem.la
tests_thirdparty_pingpong_LDADD = libshmem.la
tests_thirdparty_ptp_LDADD = libshmem.la
tests_thirdparty_put1_LDADD = libshmem.la
tests_thirdparty_put_performance_LDADD = libshmem.la
tests_thirdparty_set_lock_LDADD = libshmem.la
tests_thirdparty_shmalloc_LDADD = libshmem.la
tests_thirdparty_shmem_2dheat_LDADD = libshmem.la
tests_thirdparty_shmem_daxpy_LDADD = libshmem.la
tests_thirdparty_shmem_heat_image_LDADD = libshmem.la
tests_thirdparty_shmem_matrix_LDADD = libshmem.la
tests_thirdparty_shmemalign_LDADD = libshmem.la
tests_thirdparty_shmemlatency_LDADD = libshmem.la
tests_thirdparty_shrealloc_LDADD = libshmem.la
tests_thirdparty_spam_LDADD = libshmem.la
tests_thirdparty_sping_LDADD = libshmem.la
tests_thirdparty_strided_put_LDADD = libshmem.la
tests_thirdparty_swap1_LDADD = libshmem.la
tests_thirdparty_swapm_LDADD = libshmem.la
tests_thirdparty_to_all_LDADD = libshmem.la
tests_thirdparty_waituntil_LDADD = libshmem.la
tests_thirdparty_test_lock_LDADD = libshmem.la
tests_thirdparty_test_shmem_accessible_LDADD = libshmem.la
tests_thirdparty_test_shmem_atomics_LDADD = libshmem.la
tests_thirdparty_test_shmem_barrier_LDADD = libshmem.la
tests_thirdparty_test_shmem_broadcast_LDADD = libshmem.la
tests_thirdparty_test_shmem_collects_LDADD = libshmem.la
tests_thirdparty_test_shmem_get_LDADD = libshmem.la
tests_thirdparty_test_shmem_get_globals_LDADD = libshmem.la
tests_thirdparty_test_shmem_get_shmalloc_LDADD = libshmem.la
tests_thirdparty_test_shmem_lock_LDADD = libshmem.la
tests_thirdparty_test_shmem_put_LDADD = libshmem.la
tests_thirdparty_test_shmem_put_globals_LDADD = libshmem.la
tests_thirdparty_test_shmem_put_shmalloc_LDADD = libshmem.la
tests_thirdparty_test_shmem_reduction_LDADD = libshmem.la
tests_thirdparty_test_shmem_zero_get_LDADD = libshmem.la
tests_thirdparty_test_shmem_zero_put_LDADD = libshmem.la
tests_thirdparty_test_shmem_synchronization_LDADD = libshmem.la

