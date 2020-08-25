#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
m4_include([../maint/version.m4])
AC_INIT([oshmpi-testsuite],
        OSHMPI_VERSION_m4,
        [oshmpi-users@lists.mpich.org],
        [oshmpi],
        [https://github.com/pmodels/oshmpi])

AC_CONFIG_HEADER([include/oshmpitestconf.h])
AH_TOP([/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef OSHMPITESTCONF_H_INCLUDED
#define OSHMPITESTCONF_H_INCLUDED
])
AH_BOTTOM([#endif])

VERSION=MPICH_VERSION_m4
AC_SUBST(VERSION)
AC_CONFIG_AUX_DIR(confdb)
AC_CONFIG_MACRO_DIR(confdb)

echo "Running configure for OSHMPI tests"
AM_INIT_AUTOMAKE([-Wall -Werror -Wno-portability-recursive silent-rules foreign 1.12.3 subdir-objects])

# Non-verbose make
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_ARG_WITH(shmem, [AC_HELP_STRING([--with-shmem[=DIR|default]],
                [Use the specified SHMEM installation; Header file shmem.h should be in 
                dir/include, and oshcc should be in dir/bin. Use the in-package oshmpi by default])]
                ,,with_shmem=default)

if test "X${with_shmem}" = "Xdefault" ; then
  # note that the default definition of bindir is 
  #    '${exec_prefix}/bin'
  # so even if prefix is set, exec prefix is not until
  # the very last moment (i.e., not here). 
  if test "X$exec_prefix" = "XNONE" ; then
      saveExec_prefix=$exec_prefix
      if test "X$prefix" = "XNONE" ; then
          # Use the default
          exec_prefix=$ac_default_prefix
      else
          exec_prefix=$prefix
      fi
      # Evaluate with the current setting of exec_prefix
      eval oshmpibindir=${bindir}
      exec_prefix=$saveExec_prefix
  else
      eval oshmpibindir=${bindir}
  fi

  OSHCC=$oshmpibindir/oshcc
else
  OSHCC="${with_shmem}/bin/oshcc"
fi
AC_SUBST(OSHCC)

# Running C compiler tests
PAC_PROG_CC

# not using libtool for the test suite, so no LT_INIT.  Instead, test here
# for Library programs
AC_PROG_RANLIB
AM_PROG_AR

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_RESTRICT
AC_C_INLINE

# check for attribute support
PAC_C_GNU_ATTRIBUTE

## Enable GPU tests
AC_ARG_ENABLE(cuda-test, 
[  --enable-cuda-test=[option]
                          Enable CUDA memory kind in space tests. CUDA support is required.
                          Supported options include:
                            yes - enable CUDA memory kind tests.
                            no - disable CUDA memory kind tests (default)],
                         [ enable_cuda_test=$enableval ],
                         [ enable_cuda_test=no ])
AC_MSG_CHECKING(CUDA tests support)
AC_MSG_RESULT($enable_cuda_test)

if test "$enable_cuda_test" = "yes" ; then
    # check if CUDA is supported 
    have_cuda=no
    PAC_SET_HEADER_LIB_PATH(cuda)
    PAC_PUSH_FLAG(LIBS)
    PAC_CHECK_HEADER_LIB([cuda_runtime_api.h],[cudart],[cudaMalloc],[have_cuda=yes],[have_cuda=no])
    if test "${have_cuda}" = "yes" ; then
        AS_IF([test -d "${with_cuda}/lib64"],
            [PAC_APPEND_FLAG([-L${with_cuda}/lib64 -Wl,-rpath -Wl,${with_cuda}/lib64],[WRAPPER_LDFLAGS])])
        AS_IF([test -d "${with_cuda}/lib"],
            [PAC_APPEND_FLAG([-L${with_cuda}/lib -Wl,-rpath -Wl,${with_cuda}/lib64],[WRAPPER_LDFLAGS])])
        PAC_APPEND_FLAG([-lcuda -lcudart],[WRAPPER_LIBS])

        AC_DEFINE(OSHMPI_ENABLE_CUDA_TEST, 1, [Enable CUDA memory kind tests])
    else
        enable_cuda_test=no
        AC_MSG_ERROR(['cuda_runtime_api.h or libcudart library not found.'])
    fi
    PAC_POP_FLAG(LIBS)
fi
AM_CONDITIONAL([OSHMPI_ENABLE_CUDA_TEST], [test "$enable_cuda_test" == "yes" ])

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
