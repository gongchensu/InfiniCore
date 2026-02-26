// FlashAttention v2 style CUDA launcher for InfiniCore.
// This is a GPU implementation that will be called from the ninetoothed
// flash_attention Descriptor and replaces the original generated kernel.

#pragma once

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

// Host-side launcher. Implemented in flash_attention_v2_nvidia.cu.
// Q, K, V, O are expected to be on the same CUDA device with layout
// [B, H, S, D] and contiguous in the last dimension.
// total_kv_len is a length tensor of shape [B] (int32 / int64), currently
// only the first element is used to match existing behavior.
//
// scale: attention scaling (already includes 1/sqrt(d)).
// is_causal: whether to apply causal mask.
//
// This function must be callable from regular C++ code (Descriptor::calculate),
// so we keep the signature free of any CUDA kernel launch syntax.
infiniStatus_t flash_attention_v2_cuda_launcher(
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    const uint64_t *q_shape,
    const int64_t *q_strides,
    const uint64_t *k_shape,
    const int64_t *k_strides,
    const uint64_t *v_shape,
    const int64_t *v_strides,
    const uint64_t *len_shape,
    const int64_t *len_strides,
    infiniDtype_t dtype,
    double scale,
    char is_causal,
    cudaStream_t stream);

