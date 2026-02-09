/**
 * 海光 DCU GEMM 算子 - 使用 hipBLASLt 厂商库
 *
 * DTK hipBLASLt 库参考：ROCm hipBLASLt API
 * 支持 FP16/BF16/FP32，支持 batch GEMM
 */
#include "gemm_hygon.h"
#include "../../../../utils.h"
#include "../info.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#define CHECK_HIPBLASLT(API) CHECK_INTERNAL(API, HIPBLAS_STATUS_SUCCESS)

namespace op::gemm::hygon {

struct Descriptor::Opaque {
    hipblasLtHandle_t lt_handle;
    hipblasLtMatmulDesc_t matmul_desc;
    hipblasLtMatrixLayout_t a_layout;
    hipblasLtMatrixLayout_t b_layout;
    hipblasLtMatrixLayout_t c_layout;
    hipblasLtMatrixLayout_t d_layout;
    hipblasLtMatmulAlgo_t algo;
    bool use_heuristic;
};

Descriptor::~Descriptor() {
    if (_opaque) {
        hipblasLtMatrixLayoutDestroy(_opaque->a_layout);
        hipblasLtMatrixLayoutDestroy(_opaque->b_layout);
        hipblasLtMatrixLayoutDestroy(_opaque->c_layout);
        hipblasLtMatrixLayoutDestroy(_opaque->d_layout);
        hipblasLtMatmulDescDestroy(_opaque->matmul_desc);
        hipblasLtDestroy(_opaque->lt_handle);
        delete _opaque;
        _opaque = nullptr;
    }
}

static hipDataType toHipDataType(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        return HIP_R_16F;
    case INFINI_DTYPE_BF16:
        return HIP_R_16BF;
    case INFINI_DTYPE_F32:
        return HIP_R_32F;
    default:
        return HIP_R_32F;
    }
}

static hipblasComputeType_t toHipBlasComputeType(infiniDtype_t dtype) {
    (void)dtype;
    return HIPBLAS_COMPUTE_32F;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    (void)handle_;
    auto dtype = c_desc->dtype();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::COL_MAJOR);
    CHECK_RESULT(result);

    auto info = result.take();

    hipblasLtHandle_t lt_handle = nullptr;
    hipblasLtMatmulDesc_t matmul_desc = nullptr;
    hipblasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr;
    hipblasLtMatrixLayout_t c_layout = nullptr, d_layout = nullptr;

    CHECK_HIPBLASLT(hipblasLtCreate(&lt_handle));
    CHECK_HIPBLASLT(hipblasLtMatmulDescCreate(&matmul_desc,
                                              toHipBlasComputeType(dtype),
                                              HIP_R_32F));

    hipblasOperation_t op_a = info.a_matrix.row_stride == 1 ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    hipblasOperation_t op_b = info.b_matrix.row_stride == 1 ? HIPBLAS_OP_N : HIPBLAS_OP_T;
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(int32_t));
    hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(int32_t));

    hipDataType data_type = toHipDataType(dtype);

    uint64_t a_rows = (op_a == HIPBLAS_OP_N) ? info.m : info.k;
    uint64_t a_cols = (op_a == HIPBLAS_OP_N) ? info.k : info.m;
    int64_t a_ld = static_cast<int64_t>(info.a_matrix.ld());

    uint64_t b_rows = (op_b == HIPBLAS_OP_N) ? info.k : info.n;
    uint64_t b_cols = (op_b == HIPBLAS_OP_N) ? info.n : info.k;
    int64_t b_ld = static_cast<int64_t>(info.b_matrix.ld());

    uint64_t c_rows = info.m;
    uint64_t c_cols = info.n;
    int64_t c_ld = static_cast<int64_t>(info.c_matrix.ld());

    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&a_layout, data_type, a_rows, a_cols, a_ld));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&b_layout, data_type, b_rows, b_cols, b_ld));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&c_layout, data_type, c_rows, c_cols, c_ld));
    CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&d_layout, data_type, c_rows, c_cols, c_ld));

    if (info.batch > 1) {
        int32_t batch_count = static_cast<int32_t>(info.batch);
        int64_t a_stride = static_cast<int64_t>(info.a_matrix.stride);
        int64_t b_stride = static_cast<int64_t>(info.b_matrix.stride);
        int64_t c_stride = static_cast<int64_t>(info.c_matrix.stride);

        hipblasLtMatrixLayoutSetAttribute(a_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                          &batch_count,
                                          sizeof(batch_count));
        hipblasLtMatrixLayoutSetAttribute(a_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          &a_stride,
                                          sizeof(a_stride));

        hipblasLtMatrixLayoutSetAttribute(b_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                          &batch_count,
                                          sizeof(batch_count));
        hipblasLtMatrixLayoutSetAttribute(b_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          &b_stride,
                                          sizeof(b_stride));

        hipblasLtMatrixLayoutSetAttribute(c_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                          &batch_count,
                                          sizeof(batch_count));
        hipblasLtMatrixLayoutSetAttribute(c_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          &c_stride,
                                          sizeof(c_stride));

        hipblasLtMatrixLayoutSetAttribute(d_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                          &batch_count,
                                          sizeof(batch_count));
        hipblasLtMatrixLayoutSetAttribute(d_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          &c_stride,
                                          sizeof(c_stride));
    }

    size_t workspace_size = 0;
    hipblasLtMatmulAlgo_t algo = {};
    bool use_heuristic = false;

    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS) {
        uint64_t max_workspace = 32 * 1024 * 1024;
        hipblasLtMatmulPreferenceSetAttribute(pref,
                                              HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace,
                                              sizeof(max_workspace));

        hipblasLtMatmulHeuristicResult_t heuristic_result;
        int return_count = 0;
        hipblasStatus_t h_status = hipblasLtMatmulAlgoGetHeuristic(
            lt_handle, matmul_desc, a_layout, b_layout, c_layout, d_layout,
            pref, 1, &heuristic_result, &return_count);

        hipblasLtMatmulPreferenceDestroy(pref);

        if (h_status == HIPBLAS_STATUS_SUCCESS && return_count > 0 &&
            heuristic_result.state == HIPBLAS_STATUS_SUCCESS) {
            algo = heuristic_result.algo;
            workspace_size = heuristic_result.workspaceSize;
            use_heuristic = true;
        }
    }

    auto *opaque = new Opaque{
        lt_handle,
        matmul_desc,
        a_layout,
        b_layout,
        c_layout,
        d_layout,
        algo,
        use_heuristic};

    *desc_ptr = new Descriptor(
        dtype, info, workspace_size,
        opaque,
        INFINI_DEVICE_HYGON,
        0);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    if (_info.is_transed) {
        std::swap(a, b);
    }

    hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);

    const hipblasLtMatmulAlgo_t *algo_ptr =
        _opaque->use_heuristic ? &_opaque->algo : nullptr;

    hipblasStatus_t status = hipblasLtMatmul(
        _opaque->lt_handle,
        _opaque->matmul_desc,
        &alpha,
        a, _opaque->a_layout,
        b, _opaque->b_layout,
        &beta,
        c, _opaque->c_layout,
        c, _opaque->d_layout,
        algo_ptr,
        workspace,
        workspace_size,
        hip_stream);

    if (status != HIPBLAS_STATUS_SUCCESS) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::hygon
