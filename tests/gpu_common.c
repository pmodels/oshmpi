/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include "gpu_common.h"

#ifdef USE_CUDA
void init_device(int mype, void **device_handle)
{
    int dev_id = 0, dev_count = 0;

    cudaGetDeviceCount(&dev_count);
    dev_id = mype % dev_count;
    cudaSetDevice(dev_id);
    fprintf(stdout, "PE %d cudaSetDevice %d\n", mype, dev_id);
    fflush(stdout);
}

void reset_data(int mype, int size, int iter, int *src, int *dst)
{
    int *tmpbuf = malloc(size * iter * sizeof(int));

    for (int i = 0; i < size * iter; i++)
        tmpbuf[i] = mype + i;
    cudaMemcpy(src, tmpbuf, size * iter * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dst, 0, size * iter * sizeof(int));

    free(tmpbuf);
}

int check_data(int size, int iter, int *dst)
{
    int errs = 0;
    int *tmpbuf = malloc(size * iter * sizeof(int));

    cudaMemcpy(tmpbuf, dst, size * iter * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size * iter; i++) {
        if (tmpbuf[i] != i) {
            fprintf(stderr, "Excepted %d at dst[%d], but %d\n", i, i, tmpbuf[i]);
            fflush(stderr);
            errs++;
        }
    }

    free(tmpbuf);

    return errs;
}

#elif defined(USE_ZE)
#include <assert.h>

ze_command_list_handle_t command_list;

void init_device(int mype, void **device_handle)
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

    /* Create synchronous command list */
    ze_command_queue_desc_t descriptor;
    descriptor.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    descriptor.pNext = NULL;
    descriptor.flags = 0;
    descriptor.index = 0;
    descriptor.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    descriptor.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    uint32_t numQueueGroups = 0;
    ret = zeDeviceGetCommandQueueGroupProperties(*device_handle, &numQueueGroups, NULL);
    assert(ret == ZE_RESULT_SUCCESS && numQueueGroups > 0);
    ze_command_queue_group_properties_t *queueProperties =
        malloc(sizeof(ze_command_queue_group_properties_t) * numQueueGroups);
    ret = zeDeviceGetCommandQueueGroupProperties(*device_handle, &numQueueGroups, queueProperties);
    assert(ret == ZE_RESULT_SUCCESS);
    descriptor.ordinal = -1;
    for (int i = 0; i < numQueueGroups; i++) {
        if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            descriptor.ordinal = i;
            break;
        }
    }
    assert(descriptor.ordinal != -1);

    ret = zeCommandListCreateImmediate(ze_context, *device_handle, &descriptor, &command_list);
    assert(ret == ZE_RESULT_SUCCESS);
}

void reset_data(int mype, int size, int iter, int *src, int *dst)
{
    int *tmpbuf = malloc(size * iter * sizeof(int));

    for (int i = 0; i < size * iter; i++)
        tmpbuf[i] = mype + i;
    zeCommandListAppendMemoryCopy(command_list, src, tmpbuf, size * iter * sizeof(int), NULL, 0,
                                  NULL);
    char zero = 0;
    zeCommandListAppendMemoryFill(command_list, dst, &zero, sizeof(char), size * iter * sizeof(int),
                                  NULL, 0, NULL);
}

int check_data(int size, int iter, int *dst)
{
    int errs = 0;
    int *tmpbuf = malloc(size * iter * sizeof(int));

    zeCommandListAppendMemoryCopy(command_list, tmpbuf, dst, size * iter * sizeof(int), NULL, 0,
                                  NULL);
    for (int i = 0; i < size * iter; i++) {
        if (tmpbuf[i] != i) {
            fprintf(stderr, "Excepted %d at dst[%d], but %d\n", i, i, tmpbuf[i]);
            fflush(stderr);
            errs++;
        }
    }

    free(tmpbuf);

    return errs;
}
#else

void init_device(int mype, void **device_handle)
{
    return;
}

void reset_data(int mype, int size, int iter, int *src, int *dst)
{
    for (int i = 0; i < size * iter; i++) {
        src[i] = mype + i;
        dst[i] = 0;
    }
}

int check_data(int size, int iter, int *dst)
{
    int errs = 0;

    for (int i = 0; i < size * iter; i++) {
        if (dst[i] != i) {
            fprintf(stderr, "Excepted %d at dst[%d], but %d\n", i, i, dst[i]);
            fflush(stderr);
            errs++;
        }
    }

    return errs;
}
#endif /* end of USE_CUDA */
