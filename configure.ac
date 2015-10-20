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
AM_PROG_CC_C_O

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
AC_CHECK_LIB([mpi], [MPI_Win_allocate_shared])

# Checks for header files.
AC_CHECK_HEADERS([fcntl.h limits.h stddef.h stdio.h stdint.h stdlib.h string.h strings.h sys/param.h sys/time.h unistd.h complex.h assert.h mpi.h])

# Operating system check
AS_CASE([$host],
   [*-*-darwin*],
       [host_os=osx],
   [*-*-linux*],
       [host_os=linux],
   [AC_MSG_ERROR([Unsupported operating system.])]
   )
# set the appropriate macro
AS_CASE([$host_os],
    [osx],
        [AC_DEFINE(HAVE_APPLE_MAC,1,[Define on Apple OSX.])],
    [linux],
        [AC_DEFINE(HAVE_LINUX,1,[Define on Linux.])],
    [AC_MSG_ERROR([Unsupported operating system.])]
)

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
AC_CHECK_TYPES([ptrdiff_t])
AC_CHECK_TYPES([intptr_t])

# Checks for library functions.
#AC_FUNC_MALLOC
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

#
# OSHMPI-specific feature control
#

# SMP opts
AC_ARG_ENABLE(smp-optimizations,
              AC_HELP_STRING([--enable-smp-optimizations],[Enable SMP optimizations (i.e. MPI bypass)]),
              [ smp_optimizations_enabled=yes ],
              [ smp_optimizations_enabled=no ])
AC_MSG_CHECKING(whether SMP optimizations (i.e. MPI bypass) are enabled)
AC_MSG_RESULT($smp_optimizations_enabled)
if test "$smp_optimizations_enabled" = "yes"; then
   AC_DEFINE(ENABLE_SMP_OPTIMIZATIONS,1,[Defined when whether SMP optimizations (i.e. MPI bypass) are enabled])
fi

# MPMD
AC_ARG_ENABLE(mpmd-support,
              AC_HELP_STRING([--enable-mpmd-support],[Enable MPMD support]),
              [ mpmd_support_enabled=yes ],
              [ mpmd_support_enabled=no ])
AC_MSG_CHECKING(whether MPMD support is enabled)
AC_MSG_RESULT($mpmd_support_enabled)
if test "$mpmd_support_enabled" = "yes"; then
   AC_DEFINE(ENABLE_MPMD_SUPPORT,1,[Defined when MPMD support is enabled])
fi

# SHMEM barrier memory management routines
AC_ARG_ENABLE(sheap-barriers,
              AC_HELP_STRING([--enable-sheap-barriers],[Enable barriers in SHMEM memory management routines]),
              [ sheap_barriers_enabled=yes ],
              [ sheap_barriers_enabled=no ])
AC_MSG_CHECKING(whether SHMEM memory management routine barriers are enabled)
AC_MSG_RESULT($sheap_barriers_enabled)
if test "$sheap_barriers_enabled" = "yes"; then
   AC_DEFINE(ENABLE_SHEAP_BARRIERS,1,[Defined when SHMEM memory management routine barriers are enabled])
fi

# MPICH Ch3 is always in-order so no need to pay extra cost to order RMA.
AC_ARG_ENABLE(rma-ordering,
              AC_HELP_STRING([--enable-rma-ordering],[Enable ordering of RMA.]),
              [ rma_ordering_enabled=yes ],
              [ rma_ordering_enabled=no ])
AC_MSG_CHECKING(whether RMA ordering is enabled)
AC_MSG_RESULT($rma_ordering_enabled)
if test "$rma_ordering_enabled" = "yes"; then
   AC_DEFINE(ENABLE_RMA_ORDERING,1,[Defined when RMA ordering is enabled])
fi

## MPI subcommunicator caching
AC_ARG_ENABLE(comm-caching,
              AC_HELP_STRING([--enable-comm-caching],[Enable caching of MPI subcommunicators]),
              [ comm_caching_enabled=yes ],
              [ comm_caching_enabled=no ])
AC_MSG_CHECKING(whether communicator caching is enabled)
AC_MSG_RESULT($comm_caching_enabled)
if test "$comm_caching_enabled" = "yes"; then
   AC_DEFINE(ENABLE_COMM_CACHING,1,[Defined when MPI subcommunicator caching is enabled])
fi

#
# OpenSHMEM extensions
#

# Provide non-standard behavior w.r.t. OpenSHMEM, either to support
# other implementations (e.g. CraySHMEM), proposed extensions to OpenSHMEM,
# or OSHMPI-specific extensions that may or may not be proposed.
AC_ARG_ENABLE([extensions],
[  --enable-extensions=list - Selectively enable extensions to the OpenSHMEM
			      specification via a comma-separated list:
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
],[],[enable_extensions=none])
# strip off multiple options, separated by commas
save_IFS="$IFS"
IFS=","
for var in $enable_extensions ; do
    AS_CASE(["$var"],
            [cray_init],[enable_extension_cray_init=yes],
            [final_abort],[enable_extension_final_abort=yes],
            [ornl_aset],[enable_extension_ornl_aset=yes],
            [ornl_nbcoll],[enable_extension_ornl_nbcoll=yes],
            [ornl_nbrma],[enable_extension_ornl_nbrma=yes],
            [counting_put],[enable_extension_counting_put=yes],
            [intel_contexts],[enable_extension_intel_contexts=yes],
            [cray_threads],[enable_extension_cray_threads=yes],
            [armci_strided],[enable_extension_armci_strided=yes],
            [init_subcomm],[enable_extension_init_subcomm=yes],
            [no|none],[],
            [IFS=$save_IFS
             AC_MSG_WARN([Unknown value ($option) for enable-extensions])
             IFS=","])
done
IFS="$save_IFS"
if test -n "$enable_extension_cray_init" ; then
    AC_DEFINE(EXTENSION_CRAY_INIT,1,[Define to enable CraySHMEM initialize/finalize.])
fi
if test -n "$enable_extension_final_abort" ; then
    AC_DEFINE(EXTENSION_FINAL_ABORT,1,[Define to enable ORNL finalize/abort extension.])
fi
if test -n "$enable_extension_ornl_aset" ; then
    AC_DEFINE(EXTENSION_ORNL_ASET,1,[Define to enable ORNL active-set extension.])
fi
if test -n "$enable_extension_ornl_nbcoll" ; then
    AC_DEFINE(EXTENSION_ORNL_NBCOLL,1,[Define to enable ORNL non-blocking collectives extension.])
fi
if test -n "$enable_extension_ornl_nbrma" ; then
    AC_DEFINE(EXTENSION_ORNL_NBRMA,1,[Define to enable ORNL non-blocking RMA/AMO extension.])
fi
if test -n "$enable_extension_counting_put" ; then
    AC_DEFINE(EXTENSION_COUNTING_PUT,1,[Define to enable Intel counting put extension.])
fi
if test -n "$enable_extension_intel_contexts" ; then
    AC_DEFINE(EXTENSION_INTEL_CONTEXTS,1,[Define to enable Intel contexts extension.])
fi
if test -n "$enable_extension_cray_threads" ; then
    AC_DEFINE(EXTENSION_CRAY_THREADS,1,[Define to enable Cray thread-safety extension.])
fi
if test -n "$enable_extension_armci_strided" ; then
    AC_DEFINE(EXTENSION_ARMCI_STRIDED,1,[Define to enable ARMCI block-strided extension.])
fi
if test -n "$enable_extension_init_subcomm" ; then
    AC_DEFINE(EXTENSION_INIT_SUBCOMM,1,[Define to enable MPI subcommunicator ensemble extension.])
fi
# For easy copy-and-paste definition of new extensions.
#if test -n "$enable_extension_" ; then
#    AC_DEFINE(EXTENSION_,1,[Define to enable ])
#fi

AC_OUTPUT(Makefile)
