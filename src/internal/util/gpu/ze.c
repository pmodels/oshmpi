/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2021 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include "oshmpi_util.h"
#include <level_zero/ze_api.h>

#define ZE_ERR_CHECK(_result) \
    OSHMPI_ASSERT((_result) == ZE_RESULT_SUCCESS)

ze_driver_handle_t global_ze_driver_handle;
ze_device_handle_t *global_ze_device_handles;
ze_context_handle_t global_ze_context;
int global_ze_device_count;

void OSHMPIU_gpu_init(void)
{
    uint32_t driver_count = 0;
    ze_result_t ret;
    ze_driver_handle_t *all_drivers = NULL;

    ze_init_flag_t flags = ZE_INIT_FLAG_GPU_ONLY;
    ret = zeInit(flags);
    ZE_ERR_CHECK(ret);

    ret = zeDriverGet(&driver_count, NULL);
    ZE_ERR_CHECK(ret);
    if (driver_count == 0)
        OSHMPI_ERR_ABORT("No Level Zero device drivers found\n");

    all_drivers = OSHMPIU_malloc(driver_count * sizeof(ze_driver_handle_t));
    OSHMPI_ASSERT(all_drivers);
    ret = zeDriverGet(&driver_count, all_drivers);
    ZE_ERR_CHECK(ret);

    /* Find a driver instance with a GPU device */
    for (int i = 0; i < driver_count; ++i) {
        global_ze_device_count = 0;
        ret = zeDeviceGet(all_drivers[i], &global_ze_device_count, NULL);
        ZE_ERR_CHECK(ret);
        global_ze_device_handles =
            OSHMPIU_malloc(global_ze_device_count * sizeof(ze_device_handle_t));
        OSHMPI_ASSERT(global_ze_device_handles);
        ret = zeDeviceGet(all_drivers[i], &global_ze_device_count, global_ze_device_handles);
        ZE_ERR_CHECK(ret);
        /* Check if the driver supports a gpu */
        for (int d = 0; d < global_ze_device_count; ++d) {
            ze_device_properties_t device_properties;
            ret = zeDeviceGetProperties(global_ze_device_handles[d], &device_properties);
            ZE_ERR_CHECK(ret);

            if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                global_ze_driver_handle = all_drivers[i];
                break;
            }
        }

        if (NULL != global_ze_driver_handle) {
            break;
        }
    }

    ze_context_desc_t contextDesc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        .pNext = NULL,
        .flags = 0,
    };
    ret = zeContextCreate(global_ze_driver_handle, &contextDesc, &global_ze_context);
    ZE_ERR_CHECK(ret);

    OSHMPIU_free(all_drivers);
}

void OSHMPIU_gpu_finalize(void)
{
    OSHMPIU_free(global_ze_device_handles);
}
