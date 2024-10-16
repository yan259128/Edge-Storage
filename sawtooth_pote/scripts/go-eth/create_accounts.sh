#!/bin/bash

source ./config.sh

PASSWORD_FILE="password.txt"

# 检查并清除旧的账户信息
for node in "${NODE_DIRS[@]}"; do
  # 创建目录
  mkdir -p $node

  # 如果 keystore 目录存在且不为空，清理旧账户信息
  if [ -d "$node/keystore" ] && [ "$(ls -A $node/keystore)" ]; then
    echo "Old accounts found in $node. Clearing old accounts..."
    rm -rf $node/keystore
  fi

  echo "Creating account for $node..."

  # 停止可能已经存在的容器（确保数据目录没有被占用）
  docker container stop geth-$node || true
#  docker container rm geth-$node || true

  # 创建新账户
  ACCOUNT_ADDRESS=$(docker run --rm -v "$(pwd)/$node:/root/.ethereum" ethereum/client-go:v1.10.25 account new --password /root/.ethereum/$PASSWORD_FILE)

  # 获取创建的账户地址
  ACCOUNT_ADDRESS=$(docker run --rm -v "$(pwd)/$node:/root/.ethereum" ethereum/client-go:v1.10.25 account list | grep -oP '(?<=\{).*?(?=\})')

  # 检查账户地址是否创建成功
  if [ -z "$ACCOUNT_ADDRESS" ]; then
    echo "Error: Could not create account for $node"
  else
    echo "Account address for $node: $ACCOUNT_ADDRESS"
    # 将账号地址保存到文件
    echo "$ACCOUNT_ADDRESS" > $node/account_address.txt
  fi
done
