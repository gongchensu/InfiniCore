#!/bin/bash

echo "=== 测试海光DCU修复 ==="

# 设置环境变量
export DTK_ROOT="/opt/dtk-25.04"

echo "1. 清理配置..."
xmake f -c
xmake clean --all

# echo "2. 重新配置..."
# xmake config --hygon-dcu=true

echo "3. 测试编译..."
python scripts/install.py --hygon-dcu=y
# xmake build infiniop-hygon

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
else
    echo "❌ 编译失败"
    exit 1
fi

echo "=== 测试完成 ==="
