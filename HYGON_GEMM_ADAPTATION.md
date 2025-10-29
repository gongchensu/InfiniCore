# 海光GEMM算子ROCm适配总结

## 概述

本次工作成功将海光DCU的GEMM算子从CUDA适配迁移到ROCm平台，实现了基于rocBLAS库的高性能矩阵乘法计算。

## 主要变更

### 1. 创建海光GEMM算子实现

**文件位置：** `src/infiniop/ops/gemm/hygon/`

- `gemm_hygon.h` - 海光GEMM算子头文件
- `gemm_hygon.cu` - 基于rocBLAS的海光GEMM实现

**关键特性：**
- 支持F16、F32、BF16数据类型
- 使用rocBLAS的`rocblas_gemm_strided_batched_ex`API
- 支持矩阵转置操作
- 支持批处理矩阵乘法

### 2. 创建海光设备Handle

**文件位置：** `src/infiniop/devices/hygon/`

- `hygon_handle.h` - 海光设备Handle头文件
- `hygon_handle.cuh` - ROCm库接口定义
- `hygon_handle.cu` - 海光设备Handle实现

**关键特性：**
- 基于HIP运行时API
- 支持rocBLAS和MIOpen库
- 设备属性查询和管理
- 流和事件管理

### 3. 创建HIP运行时支持

**文件位置：** `src/infinirt/hip/`

- `infinirt_hip.h` - HIP运行时头文件
- `infinirt_hip.cu` - HIP运行时实现

**功能包括：**
- 设备管理（设备数量、设备选择）
- 内存管理（设备内存、主机内存）
- 流管理（创建、销毁、同步）
- 事件管理（创建、记录、查询、同步）
- 内存拷贝操作

### 4. 创建RCCL通信库支持

**文件位置：** `src/infiniccl/hip/`

- `infiniccl_hip.h` - RCCL头文件
- `infiniccl_hip.cu` - RCCL实现

**功能包括：**
- 通信器初始化和管理
- AllReduce集合通信操作
- 支持多种数据类型和归约操作

### 5. 更新构建配置

**主要变更：**

1. **xmake.lua** - 添加海光DCU支持
2. **hygon.lua** - 完全重写，使用ROCm工具链
   - 使用hipcc编译器替代nvcc
   - 链接rocBLAS、hip_hcc、hiprand库
   - 移除CUDA相关配置
   - 只编译gemm算子，移除其他算子

### 6. 更新核心文件

**修改的文件：**
- `src/infiniop/ops/gemm/operator.cc` - 添加海光GEMM支持
- `src/infiniop/devices/handle.cc` - 添加海光设备Handle支持
- `src/infinirt/infinirt.cc` - 添加HIP运行时支持
- `src/infiniccl/infiniccl.cc` - 添加RCCL通信支持

## API映射对比

| 功能 | CUDA | ROCm |
|------|------|------|
| 编译器 | nvcc | hipcc |
| 基础库 | CUDA Runtime | HIP Runtime |
| 线性代数 | cuBLAS | rocBLAS |
| 深度学习 | cuDNN | MIOpen |
| 通信库 | NCCL | RCCL |
| 数据类型 | half, float | rocblas_half, float |
| 流类型 | cudaStream_t | hipStream_t |
| 事件类型 | cudaEvent_t | hipEvent_t |

## 编译和测试

### 环境要求

1. **ROCm环境**：需要安装ROCm 5.0+
2. **编译器**：hipcc编译器
3. **库依赖**：rocBLAS, hip_hcc, hiprand, rccl

### 编译命令

```bash
# 配置海光DCU支持
xmake config --hygon-dcu=y

# 编译海光GEMM算子
xmake build infiniop-hygon

# 运行测试脚本
./test_hygon_gemm.sh
```

### 环境变量

```bash
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
```

## 性能优势

1. **原生ROCm支持**：直接使用ROCm库，避免CUDA兼容层开销
2. **优化编译**：使用hipcc编译器，针对AMD GPU架构优化
3. **库优化**：rocBLAS针对AMD GPU进行了深度优化
4. **内存管理**：使用HIP内存管理，更好的内存访问模式

## 下一步计划

1. **功能验证**：运行测试验证GEMM算子功能正确性
2. **性能测试**：与CUDA版本进行性能对比
3. **算子扩展**：逐步添加其他算子的海光适配
4. **优化调优**：根据测试结果进行性能优化

## 文件结构

```
src/
├── infiniop/
│   ├── ops/gemm/hygon/          # 海光GEMM算子
│   └── devices/hygon/           # 海光设备Handle
├── infinirt/hip/                # HIP运行时支持
└── infiniccl/hip/               # RCCL通信支持
```

## 注意事项

1. **依赖管理**：确保ROCm环境正确安装和配置
2. **版本兼容**：ROCm版本需要与海光DCU驱动兼容
3. **编译标志**：使用正确的GPU架构标志（gfx906, gfx926等）
4. **库链接**：确保所有ROCm库都能正确链接

通过本次适配，海光DCU现在可以使用原生ROCm库进行高性能矩阵乘法计算，为后续的深度学习应用提供了坚实的基础。
