#ifndef INFINICCL_HIP_H_
#define INFINICCL_HIP_H_

#include "../infiniccl_impl.h"

// Windows does not support HIP
#ifdef ENABLE_HYGON_API && defined(ENABLE_CCL) && !defined(_WIN32)
INFINICCL_DEVICE_API_IMPL(hip)
#else
INFINICCL_DEVICE_API_NOOP(hip)
#endif

#endif /* INFINICCL_HIP_H_ */
