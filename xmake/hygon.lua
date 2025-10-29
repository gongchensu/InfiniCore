-- ROCm工具链配置
toolchain("hygon.toolchain")
    set_toolset("cc"  , "clang"  )
    set_toolset("cxx" , "clang++")
    set_toolset("cu"  , "hipcc")
    set_toolset("culd", "hipcc")
toolchain_end()

rule("hygon.env")
    after_load(function (target)
        -- ROCm环境配置
        print("Configuring ROCm environment for Hygon DCU")
    end)
rule_end()

target("infiniop-hygon")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("hygon.toolchain")
    add_rules("hygon.env")

    -- 海光DCU使用ROCm库
    add_links("hip_hcc", "rocblas", "hiprand")
    
    -- 添加ROCm路径支持
    local rocm_root = os.getenv("ROCM_PATH") or "/opt/rocm"
    if os.isdir(rocm_root) then
        add_includedirs(path.join(rocm_root, "include"))
        add_linkdirs(path.join(rocm_root, "lib"))
        add_linkdirs(path.join(rocm_root, "lib64"))
    end

    set_warnings("all", "error")
    add_cuflags("-Wno-error=unused-private-field")
    add_cuflags("-Wno-return-type", {force = true})  -- 抑制return语句警告
    add_cuflags("-fPIC", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")

    -- 添加海光DCU特定的编译标志
    add_cuflags("-arch=gfx906", "-arch=gfx926", "-arch=gfx928", "-arch=gfx936")
    
    -- 只编译海光DCU的gemm算子
    add_files("../src/infiniop/devices/hygon/*.cu")
    add_files("../src/infiniop/ops/gemm/hygon/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", {cxflags = {"-Wno-return-type"}})
    end
target_end()

target("infinirt-hygon")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("hygon.toolchain")
    add_rules("hygon.env")

    add_links("hip_hcc", "hiprand")
    
    -- 添加ROCm路径支持
    local rocm_root = os.getenv("ROCM_PATH") or "/opt/rocm"
    if os.isdir(rocm_root) then
        add_includedirs(path.join(rocm_root, "include"))
        add_linkdirs(path.join(rocm_root, "lib"))
        add_linkdirs(path.join(rocm_root, "lib64"))
    end

    set_warnings("all", "error")
    add_cuflags("-Wno-return-type", {force = true})  -- 抑制return语句警告
    add_cuflags("-fPIC", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")

    -- 添加海光DCU特定的编译标志
    add_cuflags("-arch=gfx906", "-arch=gfx926", "-arch=gfx928", "-arch=gfx936")
    
    add_files("../src/infinirt/hip/*.cu")
target_end()

target("infiniccl-hygon")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)

    if has_config("ccl") then
        set_toolchains("hygon.toolchain")
        add_rules("hygon.env")

        add_links("hip_hcc", "hiprand")
        
        -- 添加ROCm路径支持
        local rocm_root = os.getenv("ROCM_PATH") or "/opt/rocm"
        if os.isdir(rocm_root) then
            add_includedirs(path.join(rocm_root, "include"))
            add_linkdirs(path.join(rocm_root, "lib"))
            add_linkdirs(path.join(rocm_root, "lib64"))
        end

        set_warnings("all", "error")
        add_cuflags("-Wno-return-type", {force = true})  -- 抑制return语句警告
        add_cuflags("-fPIC", "-std=c++17", {force = true})
        add_culdflags("-fPIC")
        add_cxflags("-fPIC")

        -- 添加海光DCU特定的编译标志
        add_cuflags("-arch=gfx906", "-arch=gfx926", "-arch=gfx928", "-arch=gfx936")

        -- 使用RCCL (ROCm Collective Communications Library)
        add_links("rccl")

        add_files("../src/infiniccl/hip/*.cu")
    end
target_end()
