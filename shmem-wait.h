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

#if defined(__i386__) || defined(__x86_64__)
# define SPINLOCK_BODY() do { __asm__ __volatile__ ("pause" ::: "memory"); } while (0)
#else
# define SPINLOCK_BODY() do { __asm__ __volatile__ (::: "memory"); } while (0)
#endif

#define SHMEM_WAIT(var, value)                           \
    do {                                                 \
        while (*var == value) { SPINLOCK_BODY(); }       \
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
