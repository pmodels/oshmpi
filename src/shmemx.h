#ifndef OSHMPI_SHMEMX_H
#define OSHMPI_SHMEMX_H

#include "shmemconf.h"

 /* Portals extensions */
double shmem_wtime(void);
char* shmem_nodename(void);

#if EXTENSION_FINAL_ABORT
#error TODO
#endif

#if EXTENSION_COUNTING_PUT
#error TODO
typedef char * shmemx_ct_t;

void shmemx_putmem_ct(shmem_ct_t ct, void *target, const void *source, size_t len, int pe);
void shmemx_ct_create(shmem_ct_t *ct);
void shmemx_ct_free(shmem_ct_t *ct);
long shmemx_ct_get(shmem_ct_t ct);
void shmemx_ct_set(shmem_ct_t ct, long value);
void shmemx_ct_wait(shmem_ct_t ct, long wait_for);
#endif

#if EXTENSION_INTEL_CONTEXTS
#error TODO
typedef char * shmemx_ctx_t;

shmemx_ctx_create(int num_ctx, int hint, shmem_ctx_t ctx[]);
shmemx_ctx_destroy(int num_ctx, shmem_ctx_t ctx[]);

shmemx_ctx_fence(shmem_ctx_t ctx);
shmemx_ctx_quiet(shmem_ctx_t ctx);
shmemx_ctx_barrier_all(shmem_ctx_t ctx);
shmemx_ctx_barrier(shmem_ctx_t ctx, int PE_start, int logPE_stride, int PE_size, long * pSync);

shmemx_ctx_putmem(shmem_ctx_t ctx, void * target, const void * source, size_t len, int pe);
shmemx_ctx_getmem(shmem_ctx_t ctx, void * target, const void * source, size_t len, int pe);
/* ...AND SO FORTH */
#endif

#if EXTENSION_CRAY_INIT
#error TODO
#endif

#if EXTENSION_CRAY_THREADS
#error TODO
#endif

#if EXTENSION_ORNL_ASET
#error TODO
#endif

#if EXTENSION_ORNL_NBCOLL
#error TODO
#endif

#if EXTENSION_ORNL_NBRMA
#error TODO
#endif

#if EXTENSION_ARMCI_STRIDED
#warning This is implemented on a branch for now.
#endif

#if EXTENSION_INIT_SUBCOMM
#error TODO
#endif

#endif /* OSHMPI_SHMEMX_H */
