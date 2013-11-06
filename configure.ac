# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([OSHMPI], [0.1.0], [https://github.com/jeffhammond/oshmpi/issues], [http://github.com/jeffhammond/oshmpi])
AC_CONFIG_SRCDIR([src/shmem.c])
AC_CONFIG_HEADERS([config.h])

# From ARMCI-MPI
#AC_CONFIG_AUX_DIR(m4)
#AC_CONFIG_MACRO_DIR(m4)
AM_INIT_AUTOMAKE([-Wall -Werror foreign 1.11 color-tests parallel-tests subdir-objects])

# Checks for programs.
AC_PROG_CC(mpicc)

# Checks for libraries.
AC_CHECK_LIB([m], [fabs])
AC_CHECK_LIB([mpich], [MPI_Win_allocate_shared])
#AC_CHECK_LIB([shmem], [shmem_quiet])

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h limits.h stddef.h stdint.h stdlib.h string.h strings.h sys/param.h sys/time.h unistd.h complex.h assert.h mpi.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T
AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
#AC_FUNC_ALLOCA
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_CHECK_FUNCS([atexit floor gethostname getpagesize gettimeofday memset memcpy memmove pow sqrt strerror strrchr strtol])

AC_CONFIG_FILES([src/Makefile
                 tests/Makefile
                 tests/thirdparty/Makefile])
AC_OUTPUT
