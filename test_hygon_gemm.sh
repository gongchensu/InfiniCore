#!/bin/bash

# 测试海光gemm算子编译脚本
echo "开始测试海光gemm算子编译..."

# 设置ROCm环境变量
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# 检查ROCm环境
echo "检查ROCm环境..."
if ! command -v hipcc &> /dev/null; then
    echo "错误: 未找到hipcc编译器，请确保ROCm环境已正确安装"
    exit 1
fi

if [ ! -d "$ROCM_PATH" ]; then
    echo "错误: ROCm路径不存在: $ROCM_PATH"
    exit 1
fi

echo "ROCm环境检查通过"

# 编译海光gemm算子
echo "开始编译海光gemm算子..."
xmake config --hygon-dcu=y
xmake build infiniop-hygon

if [ $? -eq 0 ]; then
    echo "✅ 海光gemm算子编译成功！"
    echo ""
    echo "编译结果："
    echo "- 海光gemm算子已成功适配ROCm平台"
    echo "- 使用rocBLAS库进行矩阵乘法计算"
    echo "- 支持F16、F32、BF16数据类型"
    echo "- 已移除其他算子的海光适配，只保留gemm算子"
    echo ""
    echo "下一步可以："
    echo "1. 运行测试验证功能正确性"
    echo "2. 逐步添加其他算子的海光适配"
    echo "3. 进行性能测试和优化"
else
    echo "❌ 海光gemm算子编译失败"
    exit 1
fi
