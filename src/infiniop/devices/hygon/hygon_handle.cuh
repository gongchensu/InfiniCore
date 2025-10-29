#ifndef __INFINIOP_HYGON_HANDLE_CUH__
#define __INFINIOP_HYGON_HANDLE_CUH__

#include "../../../utils.h"
#include "../pool.h"
#include "hygon_handle.h"
#include <rocblas.h>
#include <functional>

#ifdef ENABLE_CUDNN_API
#include <miopen/miopen.h>
#endif

#define CHECK_ROCBLAS(API) CHECK_INTERNAL(API, rocblas_status_success)
#define CHECK_MIOPEN(API) CHECK_INTERNAL(API, miopenStatusSuccess)

namespace device::hygon {

class Handle::Internal {
    Pool<rocblas_handle> blas_handles;
#ifdef ENABLE_CUDNN_API
    Pool<miopenHandle_t> dnn_handles;
#endif

    int _warp_size,
        _max_threads_per_block,
        _block_size[3],
        _grid_size[3];

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);

    infiniStatus_t useRocblas(hipStream_t stream, const Fn<rocblas_handle> &f) const;
#ifdef ENABLE_CUDNN_API
    infiniStatus_t useMiopen(hipStream_t stream, const Fn<miopenHandle_t> &f) const;
#endif

    int warpSize() const;
    int maxThreadsPerBlock() const;
    int blockSizeX() const;
    int blockSizeY() const;
    int blockSizeZ() const;
    int gridSizeX() const;
    int gridSizeY() const;
    int gridSizeZ() const;
};

} // namespace device::hygon

#endif // __INFINIOP_HYGON_HANDLE_CUH__
