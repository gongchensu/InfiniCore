#ifndef __INFINIRT_HIP_H__
#define __INFINIRT_HIP_H__
#include "../infinirt_impl.h"

namespace infinirt::hip {
#ifdef ENABLE_HYGON_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::hip

#endif // __INFINIRT_HIP_H__
