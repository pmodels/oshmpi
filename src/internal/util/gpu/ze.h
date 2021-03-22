/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef INTERNAL_UTIL_ZE_H
#define INTERNAL_UTIL_ZE_H

#include <level_zero/ze_api.h>

extern ze_context_handle_t global_ze_context;

OSHMPI_STATIC_INLINE_PREFIX OSHMPIU_gpu_pointer_type_t OSHMPIU_gpu_query_pointer_type(const void
                                                                                      *ptr)
{
    ze_result_t ret;
    ze_memory_allocation_properties_t ptr_attr;
    ze_device_handle_t device;
    OSHMPIU_gpu_pointer_type_t type = OSHMPIU_GPU_POINTER_UNREGISTERED_HOST;

    memset(&ptr_attr, 0, sizeof(ze_memory_allocation_properties_t));
    ret = zeMemGetAllocProperties(global_ze_context, ptr, &ptr_attr, &device);
    if (OSHMPI_LIKELY(ret == ZE_RESULT_SUCCESS)) {
        switch (ptr_attr.type) {
        case ZE_MEMORY_TYPE_UNKNOWN:
            type = OSHMPIU_GPU_POINTER_UNREGISTERED_HOST;
            break;
        case ZE_MEMORY_TYPE_HOST:
            type = OSHMPIU_GPU_POINTER_REGISTERED_HOST;
            break;
        case ZE_MEMORY_TYPE_DEVICE:
            type = OSHMPIU_GPU_POINTER_DEV;
            break;
        case ZE_MEMORY_TYPE_SHARED:
            type = OSHMPIU_GPU_POINTER_MANAGED;
            break;
        default:
            OSHMPI_ERR_ABORT("zeMemGetAllocProperties unsupported memory type\n");
        }
    } else {
        OSHMPI_ERR_ABORT("zeMemGetAllocProperties failed, ret %d\n", ret);
    }

    return type;
}
#endif /* INTERNAL_UTIL_ZE_H */
