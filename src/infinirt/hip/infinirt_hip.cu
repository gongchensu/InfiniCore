#include "../../utils.h"
#include "infinirt_hip.cuh"
#include <hip/hip_runtime.h>

#define CHECK_HIP(RT_API) CHECK_INTERNAL(RT_API, hipSuccess)

namespace infinirt::hip {
infiniStatus_t getDeviceCount(int *count) {
    CHECK_HIP(hipGetDeviceCount(count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_HIP(hipSetDevice(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    CHECK_HIP(hipDeviceSynchronize());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    *stream_ptr = stream;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    CHECK_HIP(hipStreamDestroy((hipStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_HIP(hipStreamSynchronize((hipStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    CHECK_HIP(hipStreamWaitEvent((hipStream_t)stream, (hipEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    hipEvent_t event;
    CHECK_HIP(hipEventCreate(&event));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_HIP(hipEventRecord((hipEvent_t)event, (hipStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    auto status = hipEventQuery((hipEvent_t)event);
    if (status == hipSuccess) {
        *status_ptr = INFINIRT_EVENT_COMPLETE;
    } else if (status == hipErrorNotReady) {
        *status_ptr = INFINIRT_EVENT_NOT_READY;
    } else {
        CHECK_HIP(status);
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_HIP(hipEventSynchronize((hipEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    CHECK_HIP(hipEventDestroy((hipEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_HIP(hipMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_HIP(hipHostMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    CHECK_HIP(hipFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    CHECK_HIP(hipHostFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

hipMemcpyKind toHipMemcpyKind(infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        return hipMemcpyHostToDevice;
    case INFINIRT_MEMCPY_D2H:
        return hipMemcpyDeviceToHost;
    case INFINIRT_MEMCPY_D2D:
        return hipMemcpyDeviceToDevice;
    case INFINIRT_MEMCPY_H2H:
        return hipMemcpyHostToHost;
    default:
        return hipMemcpyDefault;
    }
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    CHECK_HIP(hipMemcpy(dst, src, size, toHipMemcpyKind(kind)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    CHECK_HIP(hipMemcpyAsync(dst, src, size, toHipMemcpyKind(kind), (hipStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    CHECK_HIP(hipMallocAsync(p_ptr, size, (hipStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    CHECK_HIP(hipFreeAsync(ptr, (hipStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}
} // namespace infinirt::hip
