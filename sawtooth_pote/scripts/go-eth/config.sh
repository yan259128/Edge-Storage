#!/bin/bash

# 定义节点目录前缀
NODE_PREFIX="node"

# 定义节点数量
NODE_COUNT=12

# 初始化 NODE_DIRS 数组
NODE_DIRS=()

# 循环生成节点目录名称
for ((i=1; i<=NODE_COUNT; i++)); do
  NODE_DIRS+=("${NODE_PREFIX}${i}")
done