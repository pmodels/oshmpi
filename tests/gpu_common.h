/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */
#ifndef GPU_COMMON_H_INCLUDED
#define GPU_COMMON_H_INCLUDED

#ifdef USE_CUDA
#include <cuda_runtime_api.h>

static void init_device(int mype, void **device_handle)
{
    int dev_id = 0, dev_count = 0;

    cudaGetDeviceCount(&dev_count);
    dev_id = mype % dev_count;
    cudaSetDevice(dev_id);
    fprintf(stdout, "PE %d cudaSetDevice %d\n", mype, dev_id);
    fflush(stdout);
}
#elif defined(USE_ZE)
#include <level_zero/ze_api.h>
#include <assert.h>

static void init_device(int mype, void **device_handle)
{
    uint32_t driver_count = 0;
    ze_result_t ret;
    ze_driver_handle_t *all_drivers = NULL;

    ze_init_flag_t flags = ZE_INIT_FLAG_GPU_ONLY;
    ret = zeInit(flags);
    assert(ret == ZE_RESULT_SUCCESS);

    ret = zeDriverGet(&driver_count, NULL);
    assert(ret == ZE_RESULT_SUCCESS && driver_count > 0);

    all_drivers = malloc(driver_count * sizeof(ze_driver_handle_t));
    assert(all_drivers != NULL);
    ret = zeDriverGet(&driver_count, all_drivers);
    assert(ret == ZE_RESULT_SUCCESS);

    /* Find a driver instance with a GPU device */
    ze_driver_handle_t ze_driver_handle;
    ze_device_handle_t *ze_device_handles = NULL;
    int ze_device_count;
    for (int i = 0; i < driver_count; ++i) {
        ze_device_count = 0;
        ret = zeDeviceGet(all_drivers[i], &ze_device_count, NULL);
	assert(ret == ZE_RESULT_SUCCESS);
        ze_device_handles = malloc(ze_device_count * sizeof(ze_device_handle_t));
        assert(ze_device_handles != NULL);
        ret = zeDeviceGet(all_drivers[i], &ze_device_count, ze_device_handles);
	assert(ret == ZE_RESULT_SUCCESS);
        /* Check if the driver supports a gpu */
        for (int d = 0; d < ze_device_count; ++d) {
            ze_device_properties_t device_properties;
            ret = zeDeviceGetProperties(ze_device_handles[d], &device_properties);
            assert(ret == ZE_RESULT_SUCCESS);

            if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                ze_driver_handle = all_drivers[i];
                break;
            }
        }

        if (NULL != ze_driver_handle) {
            break;
        } else {
            free(ze_device_handles);
            ze_device_handles = NULL;
        }
    }

    ze_context_desc_t contextDesc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        .pNext = NULL,
        .flags = 0,
    };
    ze_context_handle_t ze_context;
    ret = zeContextCreate(ze_driver_handle, &contextDesc, &ze_context);
    assert(ret == ZE_RESULT_SUCCESS);

    *device_handle = ze_device_handles[mype % ze_device_count];

    free(ze_device_handles);
    free(all_drivers);
}
#else
static void init_device(int mype, void **device_handle)
{
    return;
}
#endif

#endif /* GPU_COMMON_H_INCLUDED */
