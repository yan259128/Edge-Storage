#!/bin/bash

source ./config.sh

GENESIS_FILE="./genesis.json"

# 初始化每个节点
for node in "${NODE_DIRS[@]}"; do
  mkdir -p $node
  echo "Cleaning old data for $node..."

  # 清理旧的区块链数据
  rm -rf $node/geth $node/chaindata $node/lightchaindata

  echo "Initializing $node..."

  # 使用 docker run 初始化 Geth 节点
  docker run --rm -v "$(pwd)/$node:/root/.ethereum" -v "$(pwd)/$GENESIS_FILE:/root/genesis.json" ethereum/client-go:v1.10.25 init /root/genesis.json
done
