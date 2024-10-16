#!/bin/bash

# 定义集群节点数，可以根据需要调整
NODE_COUNT=12  # 集群中的节点数量，动态扩展时可以增加这个数字

# 启动所有节点
docker-compose up -d

# 等待节点启动
echo "Waiting for nodes to start..."
sleep 10  # 可以根据需要调整等待时间

# 存储所有节点的 enode 信息
declare -A ENODES

# 获取所有节点的 enode 信息
for i in $(seq 1 $NODE_COUNT); do
  NODE_NAME="geth-node$i"
  HTTP_PORT=$((8544 + $i))  # 动态分配HTTP端口，确保不冲突
  NODE_HTTP="http://localhost:$HTTP_PORT"

  # 获取容器的 IP 地址
  NODE_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $NODE_NAME)

  # 获取 enode 地址
  echo "Fetching enode for $NODE_NAME..."
  ENODE=$(geth attach $NODE_HTTP --exec "admin.nodeInfo.enode" | tr -d '"')

  # 替换 enode 中的 127.0.0.1 为容器的实际 IP 地址
  ENODE_WITH_IP=${ENODE/127.0.0.1/$NODE_IP}

  # 存储 enode 信息
  ENODES[$NODE_NAME]=$ENODE_WITH_IP
  echo "$NODE_NAME enode: ${ENODES[$NODE_NAME]}"
done

# 互相添加 peer
for i in $(seq 1 $NODE_COUNT); do
  NODE_NAME="geth-node$i"
  HTTP_PORT=$((8544 + $i))  # 动态分配HTTP端口
  NODE_HTTP="http://localhost:$HTTP_PORT"

  for j in $(seq 1 $NODE_COUNT); do
    if [ $i -ne $j ]; then
      PEER_NAME="geth-node$j"
      echo "Adding $PEER_NAME to $NODE_NAME peers..."
      geth attach $NODE_HTTP --exec "admin.addPeer('${ENODES[$PEER_NAME]}')"
    fi
  done
done

# 检查节点同步状态，所有节点同步完毕后退出
echo "Checking sync status of all nodes..."

ALL_SYNCED=false

while [ "$ALL_SYNCED" == "false" ]; do
  ALL_SYNCED=true  # 假设所有节点同步完成
  for i in $(seq 1 $NODE_COUNT); do
    NODE_NAME="geth-node$i"
    HTTP_PORT=$((8544 + $i))  # 动态分配HTTP端口
    NODE_HTTP="http://localhost:$HTTP_PORT"

    # 检查节点的同步状态
    SYNCING=$(geth attach $NODE_HTTP --exec "eth.syncing")
    if [ "$SYNCING" != "false" ]; then
      ALL_SYNCED=false  # 如果有任意一个节点仍在同步，则设置为 false
      echo "$NODE_NAME is still syncing..."
    else
      echo "$NODE_NAME is synced."
    fi
  done

  if [ "$ALL_SYNCED" == "false" ]; then
    echo "Waiting for all nodes to finish syncing..."
    sleep 5  # 每次检查同步状态之前等待 5 秒
  fi
done

echo "All nodes are fully synced. Cluster setup complete."