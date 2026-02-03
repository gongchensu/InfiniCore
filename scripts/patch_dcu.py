import argparse
import os

# Patch: fix Python extension names and C launcher API for HIP (Hygon DCU).
REPLACEMENTS = {
    # 1. Python layer: extension names (cubin -> hsaco, cuda -> hip)
    "compile.py": {
        '"cubin"': '"hsaco"', 
        "'cubin'": "'hsaco'",
        "cuda": "hip",
    },
    
    # 2. C launcher template: replace CUDA Driver API with HIP Driver API
    "compile.c": {
        # Headers
        "<cuda.h>": "<hip/hip_runtime.h>",
        
        # Functions
        "cuLaunchKernel": "hipModuleLaunchKernel",
        "cuModuleGetFunction": "hipModuleGetFunction",
        "cuModuleLoadData": "hipModuleLoadData",
        "cuModuleUnload": "hipModuleUnload",
        "cuDeviceGetAttribute": "hipDeviceGetAttribute",
        "cuGetErrorString": "hipGetErrorString",
        "cuCtxGetId": "hipGetDevice",  # Note: HIP uses GetDevice instead of context ID
        "cuInit": "hipInit",
        "cuDeviceGet": "hipDeviceGet",
        "cuCtxGetCurrent": "hipCtxGetCurrent",
        "cuStreamSynchronize": "hipStreamSynchronize",
        
        # Types
        "CUresult": "hipError_t",
        "CUstream": "hipStream_t",
        "CUfunction": "hipFunction_t",
        "CUmodule": "hipModule_t",
        "CUcontext": "hipCtx_t",
        "CUdevice": "hipDevice_t",
        
        # Constants/macros
        "CUDA_SUCCESS": "hipSuccess",
        "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN": "hipDeviceAttributeMaxSharedMemoryPerBlock",
    },
    
    # 3. compile.h (if present)
    "compile.h": {
        "<cuda.h>": "<hip/hip_runtime.h>",
        "CUresult": "hipError_t",
        "CUstream": "hipStream_t",
    }
}

def replace_in_file(file_path, replacements):
    if not os.path.exists(file_path):
        # Some Triton versions omit compile.h; skip if missing.
        if "compile.h" not in file_path: 
            print(f"Warning: File `{file_path}` does not exist, skipping.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    changed = False
    for old, new in replacements.items():
        if old in content:
            content = content.replace(old, new)
            print(f"  [{os.path.basename(file_path)}] Replaced {old} -> {new}")
            changed = True

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Success: Updated `{file_path}`.")
    else:
        print(f"Info: No changes needed for `{file_path}`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch Triton for Hygon DCU (C/Python API Fix).")
    parser.add_argument(
        "directory", type=str, help="Location of the installed triton package."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: Directory `{args.directory}` does not exist.")
        exit(1)

    print(f"Patching Triton in: {args.directory}")
    
    for filename, replacements in REPLACEMENTS.items():
        file_path = os.path.join(args.directory, "tools", filename)
        replace_in_file(file_path, replacements)
