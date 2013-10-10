/* -*- C -*-
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 * 
 * This file is part of the Portals SHMEM software package. For license
 * information, see the LICENSE file in the top level directory of the
 * distribution.
 *
 */

#ifndef SHMEM_WAIT_H
#define SHMEM_WAIT_H

#include "shmem-internals.h"

#define RAISE_ERROR(code)                                \
    __shmem_abort(code, "invalid comparison")

#if defined(__i386__) || defined(__x86_64__)
# define SPINLOCK_BODY() do { __asm__ __volatile__ ("pause" ::: "memory"); } while (0)
#else
# define SPINLOCK_BODY() do { __asm__ __volatile__ (::: "memory"); } while (0)
#endif

#define SHMEM_WAIT(var, value)                           \
    do {                                                 \
        while (*var == value) { SPINLOCK_BODY(); }       \
    } while(0)


#define COMP(type, a, b, ret)                            \
    do {                                                 \
        ret = 0;                                         \
        switch (type) {                                  \
        case SHMEM_CMP_EQ:                               \
            if (a == b) ret = 1;                         \
            break;                                       \
        case SHMEM_CMP_NE:                               \
            if (a != b) ret = 1;                         \
            break;                                       \
        case SHMEM_CMP_GT:                               \
            if (a > b) ret = 1;                          \
            break;                                       \
        case SHMEM_CMP_GE:                               \
            if (a >= b) ret = 1;                         \
            break;                                       \
        case SHMEM_CMP_LT:                               \
            if (a < b) ret = 1;                          \
            break;                                       \
        case SHMEM_CMP_LE:                               \
            if (a <= b) ret = 1;                         \
            break;                                       \
        default:                                         \
            RAISE_ERROR(-1);                             \
        }                                                \
    } while(0)


#define SHMEM_WAIT_UNTIL(var, cond, value)               \
    do {                                                 \
        int cmpret;                                      \
                                                         \
        COMP(cond, *var, value, cmpret);                 \
        while (!cmpret) {                                \
            SPINLOCK_BODY();                             \
            COMP(cond, *var, value, cmpret);             \
        }                                                \
    } while(0)


#endif // SHMEM_WAIT_H
