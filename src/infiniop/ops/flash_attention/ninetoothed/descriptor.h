#ifndef __FLASH_ATTENTION_DESCRIPTOR_H__
#define __FLASH_ATTENTION_DESCRIPTOR_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../ninetoothed/utils.h"
#include "../nvidia/flash_attention_v2_nvidia.h"

namespace op::flash_attention::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(infiniopHandle_t handle,
               infiniopTensorDescriptor_t out_desc,
               infiniopTensorDescriptor_t q_desc,
               infiniopTensorDescriptor_t k_desc,
               infiniopTensorDescriptor_t v_desc,
               infiniopTensorDescriptor_t total_kv_len,
               double scale,
               char is_causal) : InfiniopDescriptor{handle->device, handle->device_id},
                                 _query_shape{q_desc->shape()},
                                 _query_strides{q_desc->strides()},
                                 _key_shape{k_desc->shape()},
                                 _key_strides{k_desc->strides()},
                                 _value_shape{v_desc->shape()},
                                 _value_strides{v_desc->strides()},
                                 _total_kv_shape{total_kv_len->shape()},
                                 _total_kv_strides{total_kv_len->strides()},
                                 _output_strides{out_desc->strides()},
                                 _dtype{q_desc->dtype()},
                                 _scale{scale},
                                 _is_causal{is_causal} {
    }

    ~Descriptor() = default;

    size_t get_workspace_size() const {
        return 0;
    }

    infiniStatus_t calculate(void *workspace,
                             size_t workspace_size,
                             void *out,
                             const void *q,
                             const void *k,
                             const void *v,
                             const void *total_kv_len,
                             void *stream) const {
        (void)workspace;
        (void)workspace_size;

        // Directly call our FlashAttention v2 CUDA launcher.
        return flash_attention_v2_cuda_launcher(
            out,
            q,
            k,
            v,
            total_kv_len,
            _query_shape.data(),
            _query_strides.data(),
            _key_shape.data(),
            _key_strides.data(),
            _value_shape.data(),
            _value_strides.data(),
            _total_kv_shape.data(),
            _total_kv_strides.data(),
            _dtype,
            _scale,
            _is_causal,
            static_cast<cudaStream_t>(stream));
    }

    static infiniStatus_t create(infiniopHandle_t handle,
                                 Descriptor **desc,
                                 infiniopTensorDescriptor_t out_desc,
                                 infiniopTensorDescriptor_t q_desc,
                                 infiniopTensorDescriptor_t k_desc,
                                 infiniopTensorDescriptor_t v_desc,
                                 infiniopTensorDescriptor_t total_kv_len,
                                 double scale,
                                 char is_causal) {
        *desc = new Descriptor{handle, out_desc, q_desc, k_desc, v_desc, total_kv_len, scale, is_causal};

        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;

    using Stride = ::ninetoothed::Tensor<>::Stride;

    std::vector<Size> _query_shape;

    std::vector<Stride> _query_strides;

    std::vector<Size> _key_shape;

    std::vector<Stride> _key_strides;

    std::vector<Size> _value_shape;

    std::vector<Stride> _value_strides;

    std::vector<Size> _total_kv_shape;

    std::vector<Stride> _total_kv_strides;

    std::vector<Stride> _output_strides;

    infiniDtype_t _dtype;

    double _scale;

    char _is_causal;
};

} // namespace op::flash_attention::ninetoothed

#endif // __FLASH_ATTENTION_DESCRIPTOR_H__
