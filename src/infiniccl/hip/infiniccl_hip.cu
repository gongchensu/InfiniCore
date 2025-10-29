#include "infiniccl_hip.h"

#include <hip/hip_runtime.h>
#include <iostream>
#include <rccl.h>
#include <vector>

#include "../../utils.h"

#define CHECK_RCCL(API__) CHECK_INTERNAL(API__, rcclSuccess)

inline hipStream_t getHipStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<hipStream_t>(stream);
}

inline rcclDataType_t getRcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return rcclFloat;
    case INFINI_DTYPE_F16:
        return rcclHalf;
    case INFINI_DTYPE_BF16:
        return rcclBfloat16;
    default:
        std::abort();
        return rcclHalf;
    }
}

inline rcclRedOp_t getRcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return rcclSum;
    case INFINICCL_PROD:
        return rcclProd;
    case INFINICCL_MAX:
        return rcclMax;
    case INFINICCL_MIN:
        return rcclMin;
    case INFINICCL_AVG:
        return rcclAvg;
    default:
        std::abort();
        return rcclSum;
    }
}

inline rcclComm_t getRcclComm(infinicclComm_t comm) {
    return static_cast<rcclComm_t>(comm->comm);
}

namespace infiniccl::hip {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<rcclComm_t> rccl_comms(ndevice);
    CHECK_RCCL(rcclCommInitAll(rccl_comms.data(), ndevice, (int const *)device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_HYGON, device_ids[i], (void *)(rccl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_RCCL(rcclCommDestroy(getRcclComm(comm)));
    delete comm;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    CHECK_RCCL(rcclAllReduce(sendbuf, recvbuf, count, getRcclDtype(datatype),
                             getRcclRedOp(op), getRcclComm(comm), getHipStream(stream)));

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::hip
