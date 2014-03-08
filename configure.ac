# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([OSHMPI], [0.1.0], [https://github.com/jeffhammond/oshmpi/issues], [http://github.com/jeffhammond/oshmpi])
AC_CONFIG_SRCDIR([src/shmem.c])

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
AC_CHECK_HEADERS([fcntl.h limits.h stddef.h stdio.h stdint.h stdlib.h string.h strings.h sys/param.h sys/time.h unistd.h complex.h assert.h mpi.h])

AC_CONFIG_HEADER(src/shmemconf.h)
AH_TOP([/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2013 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#ifndef _SHMEMCONF_H_
#define _SHMEMCONF_H_
])
AH_BOTTOM([#endif /* _SHMEMCONF_H_ */])

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

# Check for support for weak symbols.
AC_ARG_ENABLE(weak-symbols, AC_HELP_STRING([--enable-weak-symbols],
                 [Use weak symbols to implement PSHMEM routines (default)]),,
                 enable_weak_symbols=yes)
if test $enable_weak_symbols = yes ; then
    # Turn off weak symbols if they aren't available
    PAC_PROG_C_WEAK_SYMBOLS(,enable_weak_symbols=no)
fi
if test $enable_weak_symbols = "yes" ; then
    AC_DEFINE(USE_WEAK_SYMBOLS,1,[Define if weak symbols should be used])
    # Check for the ability to support multiple weak symbols
    if test "$pac_cv_prog_c_weak_symbols" = "pragma weak" ; then
       PAC_PROG_C_MULTIPLE_WEAK_SYMBOLS(AC_DEFINE(HAVE_MULTIPLE_PRAGMA_WEAK,1,[Define if multiple weak symbols may be defined]))
    fi
fi

## SHMEM barrier memory management routines
AC_ARG_ENABLE(sheap-barriers, AC_HELP_STRING([--enable-sheap-barriers],[Enable barriers in SHMEM memory management routines]),
                 [ sheap_barriers_enabled=$enableval ],
                 [ sheap_barriers_enabled=no ])
AC_MSG_CHECKING(whether SHMEM memory management routine barriers are enabled)
AC_MSG_RESULT($sheap_barriers_enabled)
if test "$sheap_barriers_enabled" = "yes"; then
   AC_DEFINE(SHEAP_BARRIERS,1,[Defined when SHMEM memory management routine barriers are enabled])
fi

## MPI subcommunicator caching
AC_ARG_ENABLE(comm-caching, AC_HELP_STRING([--enable-comm-caching],[Enable caching of communicators]),
                 [ comm_caching_enabled=$enableval ],
                 [ comm_caching_enabled=no ])
AC_MSG_CHECKING(whether communicator caching is enabled)
AC_MSG_RESULT($comm_caching_enabled)
if test "$comm_caching_enabled" = "yes"; then
   AC_DEFINE(COMM_CACHING,1,[Defined when SHMEM communicator caching is enabled])
fi

# 
AC_ARG_ENABLE(rma-ordering,
[  --enable-rma-ordering=level
      Control the amount of error checking.  
        no        - Unordered RMA with explicit fence/quiet. (default)
        yes       - Ordered RMA with no-op fence/quiet.
],,enable_error_checking=default)

# Error checking (once implemented).
AC_ARG_ENABLE(error-checking,
[  --enable-error-checking=level
      Control the amount of error checking.  
        no        - No error checking.
        yes       - Error checking. (default)
],,enable_error_checking=default)

# Provide non-standard behavior w.r.t. OpenSHMEM, either to support
# other implementations (e.g. CraySHMEM), proposed extensions to OpenSHMEM,
# or OSHMPI-specific extensions that may or may not be proposed.
AC_ARG_ENABLE([extensions],
[  --enable-extensions=list - Selectively enable extensions to the OpenSHMEM
			      specification; list is a comma-separated module names,
			      including (default is "none"):
        none               - No extensions. (default)
        cray_init          - CraySHMEM initialize/finalize.
        final_abort        - ORNL finalize/abort extension.
        ornl_aset          - ORNL active-set extension.
        ornl_nbcoll        - ORNL non-blocking collectives extension.
        ornl_nbrma         - ORNL non-blocking RMA/AMO extension.
        counting_put       - Intel counting put extension.
        cray_threads       - Cray thread-safety extension.
        armci_strided      - ARMCI block-strided extension.
        init_subcomm       - MPI subcommunicator ensemble extension.
        all                - All variables above.
],[],[enable_extensions=none])

AC_OUTPUT(Makefile)
