#ifndef OSHMPI_SHMEMX_H
#define OSHMPI_SHMEMX_H

#include "shmemconf.h"
#include "shmem.h"

 /* Portals extensions */
double shmem_wtime(void);
char* shmem_nodename(void);

#if EXTENSION_FINAL_ABORT
#error TODO
#endif

#if EXTENSION_COUNTING_PUT
typedef long * shmemx_ct_t;

/* Collective */
void shmemx_ct_create(shmemx_ct_t *ct);
void shmemx_ct_free(shmemx_ct_t *ct);

/* Local */
long shmemx_ct_get(shmemx_ct_t ct);
void shmemx_ct_set(shmemx_ct_t ct, long value);
void shmemx_ct_wait(shmemx_ct_t ct, long wait_for);

/* P2P Communication */
void shmemx_putmem_ct(shmemx_ct_t ct, void *target, const void *source, size_t len, int pe);
#endif

#if EXTENSION_INTEL_CONTEXTS
#error TODO
typedef char * shmemx_ctx_t;

shmemx_ctx_create(int num_ctx, int hint, shmemx_ctx_t ctx[]);
shmemx_ctx_destroy(int num_ctx, shmemx_ctx_t ctx[]);

shmemx_ctx_fence(shmemx_ctx_t ctx);
shmemx_ctx_quiet(shmemx_ctx_t ctx);
shmemx_ctx_barrier_all(shmemx_ctx_t ctx);
shmemx_ctx_barrier(shmemx_ctx_t ctx, int PE_start, int logPE_stride, int PE_size, long * pSync);

shmemx_ctx_putmem(shmemx_ctx_t ctx, void * target, const void * source, size_t len, int pe);
shmemx_ctx_getmem(shmemx_ctx_t ctx, void * target, const void * source, size_t len, int pe);
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
void shmemx_aput(void *target, const void *source,
                 ptrdiff_t target_ptrdiff, ptrdiff_t source_ptrdiff, size_t len, int pe);
void shmemx_aget(void *target, const void *source,
                 ptrdiff_t target_ptrdiff, ptrdiff_t source_ptrdiff, size_t len, int pe);
#endif

#if EXTENSION_INIT_SUBCOMM
#error TODO
#endif

#endif /* OSHMPI_SHMEMX_H */
