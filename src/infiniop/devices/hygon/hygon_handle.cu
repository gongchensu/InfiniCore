#include "hygon_handle.cuh"

namespace device::hygon {

Handle::Handle(infiniDevice_t device, int device_id) : InfiniopHandle{device, device_id} {
    _internal = std::make_shared<Internal>(device_id);
}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(INFINI_DEVICE_HYGON, device_id);
    return INFINI_STATUS_SUCCESS;
}

Handle::Internal::Internal(int device_id) {
    hipSetDevice(device_id);
    
    // 获取设备属性
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id);
    
    _warp_size = prop.warpSize;
    _max_threads_per_block = prop.maxThreadsPerBlock;
    _block_size[0] = prop.maxThreadsDim[0];
    _block_size[1] = prop.maxThreadsDim[1];
    _block_size[2] = prop.maxThreadsDim[2];
    _grid_size[0] = prop.maxGridSize[0];
    _grid_size[1] = prop.maxGridSize[1];
    _grid_size[2] = prop.maxGridSize[2];
}

infiniStatus_t Handle::Internal::useRocblas(hipStream_t stream, const Fn<rocblas_handle> &f) const {
    auto handle = blas_handles.get();
    rocblas_set_stream(handle, stream);
    return f(handle);
}

#ifdef ENABLE_CUDNN_API
infiniStatus_t Handle::Internal::useMiopen(hipStream_t stream, const Fn<miopenHandle_t> &f) const {
    auto handle = dnn_handles.get();
    miopenSetStream(handle, stream);
    return f(handle);
}
#endif

int Handle::Internal::warpSize() const {
    return _warp_size;
}

int Handle::Internal::maxThreadsPerBlock() const {
    return _max_threads_per_block;
}

int Handle::Internal::blockSizeX() const {
    return _block_size[0];
}

int Handle::Internal::blockSizeY() const {
    return _block_size[1];
}

int Handle::Internal::blockSizeZ() const {
    return _block_size[2];
}

int Handle::Internal::gridSizeX() const {
    return _grid_size[0];
}

int Handle::Internal::gridSizeY() const {
    return _grid_size[1];
}

int Handle::Internal::gridSizeZ() const {
    return _grid_size[2];
}

} // namespace device::hygon
