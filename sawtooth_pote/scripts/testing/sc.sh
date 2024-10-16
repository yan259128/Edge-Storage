#!/bin/bash

# 从 rest-api-url 文件中读取主机地址
if [ -f "./rest-api-url.txt" ]; then
    REST_API_HOST=$(cat ./rest-api-url.txt)
    URL="http://$REST_API_HOST:8008"  # 自动补充协议和端口
    echo "Using REST API URL: $URL"
else
    echo "Error: rest-api-url file not found!"
    exit 1
fi

if [ -f "./username.txt" ]; then
    USERNAME=$(cat ./username.txt)
    echo "Using REST API URL: $URL"
else
    echo "Error: username file not found!"
    exit 1
fi


for ((i=1; i<=20; i++)); do
        intkey set --url $URL --keyfile /root/.sawtooth/keys/$USERNAME.priv $(date +%s%N) 2
    sleep 1
    done