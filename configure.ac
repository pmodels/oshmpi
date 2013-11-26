# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([OSHMPI], [0.1.0], [https://github.com/jeffhammond/oshmpi/issues], [http://github.com/jeffhammond/oshmpi])
AC_CONFIG_SRCDIR([src/shmem.c])
AC_CONFIG_HEADERS([config.h])

# From ARMCI-MPI
AC_CONFIG_AUX_DIR(m4)
AC_CONFIG_MACRO_DIR(m4)
AM_INIT_AUTOMAKE([-Wall -Werror foreign 1.11 color-tests parallel-tests subdir-objects])

# Checks for programs.
AC_PROG_CC

## const and restrict
AC_C_CONST
AC_C_RESTRICT

## Check for C99
AC_PROG_CC_C99
if test "$ac_cv_prog_cc_c99" = "no" ; then
  AC_ERROR([C99 not supported by the compiler])
fi

# automake 1.12 seems to require this, but automake 1.11 doesn't recognize it
# must come before LT_INIT
m4_ifdef([AM_PROG_AR],[AM_PROG_AR])

LT_INIT(disable-shared)

## Non-verbose make
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Checks for libraries.
AC_CHECK_LIB([m], [fabs])
AC_CHECK_LIB([mpich], [MPI_Win_allocate_shared])

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
AC_FUNC_MALLOC
AC_CHECK_FUNCS([atexit floor gethostname getpagesize gettimeofday memset memcpy memmove posix_memalign pow sqrt strerror strrchr strtol])

## Debugging support
AC_ARG_ENABLE(g, AC_HELP_STRING([--enable-g],[Enable Debugging]),
                                [ debug=$enableval ],
                                [ debug=no ])
AC_MSG_CHECKING(debugging support)
AC_MSG_RESULT($debug)
if test "$debug" = "yes"; then
  CFLAGS="$CFLAGS -g -O0"
fi

## Check for support for weak symbols.
#AC_ARG_ENABLE(weak-symbols, AC_HELP_STRING([--enable-weak-symbols],
#                 [Use weak symbols to implement PARMCI routines (default)]),,
#                 enable_weak_symbols=yes)
#if test $enable_weak_symbols = yes ; then
#    # Turn off weak symbols if they aren't available
#    PAC_PROG_C_WEAK_SYMBOLS(,enable_weak_symbols=no)
#fi
#if test $enable_weak_symbols = "yes" ; then
#    AC_DEFINE(USE_WEAK_SYMBOLS,1,[Define if weak symbols should be used])
#    # Check for the ability to support multiple weak symbols
#    if test "$pac_cv_prog_c_weak_symbols" = "pragma weak" ; then
#       PAC_PROG_C_MULTIPLE_WEAK_SYMBOLS(AC_DEFINE(HAVE_MULTIPLE_PRAGMA_WEAK,1,[Define if multiple weak symbols may be defined]))
#    fi
#fi

#AC_CONFIG_FILES([Makefile])

AC_OUTPUT(Makefile)
