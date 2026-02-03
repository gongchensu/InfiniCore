/* HIP compat for ninetoothed AOT code on Hygon DCU: include HIP and provide CUDA type/API shims.
 * Included via -include when compiling build/ninetoothed .c/.cpp.
 */
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>

#ifndef __CUDA_DEVICE_PTR_DEFINED
#define __CUDA_DEVICE_PTR_DEFINED
typedef hipDeviceptr_t CUdeviceptr;
#endif

/* CUDA cuGetErrorString(code, &str) was patched to hipGetErrorString(code, &str), but HIP only has
 * hipGetErrorString(hipError_t). Macro dispatches by arity: 1 arg -> call original; 2 args -> write through pointer.
 */
#define HIP_GET_ERROR_STRING_1(a) (::hipGetErrorString(a))
#define HIP_GET_ERROR_STRING_2(a, b) ((*(b)) = ::hipGetErrorString(a))
#define HIP_GET_ERROR_STRING_CHOOSER(_1, _2, _3, ...) _3
#define hipGetErrorString(...) HIP_GET_ERROR_STRING_CHOOSER(__VA_ARGS__, HIP_GET_ERROR_STRING_2, HIP_GET_ERROR_STRING_1)(__VA_ARGS__)

/* CUDA Driver API -> HIP name mapping */
/* hipCtxGetId is [[nodiscard]]; generated code ignores return; void wrapper avoids -Werror=unused-result */
#ifndef cuCtxGetId
static inline void cuCtxGetId_void(void* ctx, unsigned long long* ctxId) {
  (void)hipCtxGetId((hipCtx_t)ctx, ctxId);
}
#define cuCtxGetId cuCtxGetId_void
#endif
#ifndef cuFuncSetCacheConfig
#define cuFuncSetCacheConfig hipFuncSetCacheConfig
#endif
#ifndef cuFuncSetAttribute
#define cuFuncSetAttribute hipFuncSetAttribute
#endif

#ifndef CU_FUNC_CACHE_PREFER_SHARED
#define CU_FUNC_CACHE_PREFER_SHARED hipFuncCachePreferShared
#endif
#ifndef CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES hipFuncAttributeMaxDynamicSharedMemorySize
#endif

#endif
