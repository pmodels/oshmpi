/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2018 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_UTIL_THREAD_H
#define INTERNAL_UTIL_THREAD_H

#include "oshmpi_util.h"

/* Critical section implementation routines */
#define OSHMPI_THREAD_CS_LOCK__PTHREAD_MUTEX 1

#if defined (OSHMPI_THREAD_CS_LOCK) && (OSHMPI_THREAD_CS_LOCK == OSHMPI_THREAD_CS_LOCK__PTHREAD_MUTEX)
typedef struct {
    pthread_mutex_t mutex;
    unsigned short is_initialied;
} OSHMPIU_thread_cs_t;

OSHMPI_STATIC_INLINE_PREFIX int OSHMPIU_thread_cs_init(OSHMPIU_thread_cs_t * cs_ptr)
{
    int err = 0;
    err = pthread_mutex_init(&(cs_ptr)->mutex, NULL);
    if (err == 0)
        cs_ptr->is_initialied = 1;
    return err;
}

OSHMPI_STATIC_INLINE_PREFIX int OSHMPIU_thread_cs_destroy(OSHMPIU_thread_cs_t * cs_ptr)
{
    int err = 0;
    err = pthread_mutex_destroy(&(cs_ptr)->mutex);
    if (err == 0)
        cs_ptr->is_initialied = 0;
    return err;
}

#define OSHMPIU_THREAD_CS_ENTER(cs_ptr)  pthread_mutex_lock(&(cs_ptr)->mutex)
#define OSHMPIU_THREAD_CS_EXIT(cs_ptr)  pthread_mutex_unlock(&(cs_ptr)->mutex)
#define OSHMPIU_THREAD_CS_IS_INITIALIZED(cs_ptr)  ((cs_ptr)->is_initialied == 1)
#else /* OSHMPI_THREAD_CS_LOCK__PTHREAD_MUTEX */

typedef struct {
    int dummy;
} OSHMPIU_thread_cs_t;

OSHMPI_STATIC_INLINE_PREFIX int OSHMPIU_thread_cs_init(OSHMPIU_thread_cs_t * cs_ptr)
{
    return 0;
}

OSHMPI_STATIC_INLINE_PREFIX int OSHMPIU_thread_cs_destroy(OSHMPIU_thread_cs_t * cs_ptr)
{
    return 0;
}

#define OSHMPIU_THREAD_CS_ENTER(cs_ptr)
#define OSHMPIU_THREAD_CS_EXIT(cs_ptr)
#define OSHMPIU_THREAD_CS_IS_INITIALIZED(cs_ptr)

#endif /* OSHMPI_THREAD_CS_LOCK__PTHREAD_MUTEX */

#endif /* INTERNAL_UTIL_THREAD_H */
