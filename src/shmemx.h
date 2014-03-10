#ifndef OSHMPI_SHMEMX_H
#define OSHMPI_SHMEMX_H

#include "shmemconf.h"

 /* Portals extensions */
double shmem_wtime(void);
char* shmem_nodename(void);

#if EXTENSION_FINAL_ABORT
#endif

#if EXTENSION_COUNTING_PUT
/* Signalling puts */
typedef char * shmem_ct_t;

void shmem_putmem_ct(shmem_ct_t ct, void *target, const void *source, size_t len, int pe);
void shmem_ct_create(shmem_ct_t *ct);
void shmem_ct_free(shmem_ct_t *ct);
long shmem_ct_get(shmem_ct_t ct);
void shmem_ct_set(shmem_ct_t ct, long value);
void shmem_ct_wait(shmem_ct_t ct, long wait_for);
#endif

#if EXTENSION_CRAY_INIT
#endif

#if EXTENSION_CRAY_THREADS
#endif

#if EXTENSION_ORNL_ASET
#endif

#if EXTENSION_ORNL_NBCOLL
#endif

#if EXTENSION_ORNL_NBRMA
#endif

#if EXTENSION_ARMCI_STRIDED
#endif

#if EXTENSION_INIT_SUBCOMM
#endif

#endif /* OSHMPI_SHMEMX_H */
