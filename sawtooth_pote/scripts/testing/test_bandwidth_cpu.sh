#!/bin/bash

# 输出的 CSV 文件名
output_file="cpu_bandwidth_usage.csv"

# 清空旧的 CSV 文件，写入表头
echo "Time (s),Container ID,Container Name,CPU (%),NetIO (Sent),NetIO (Received)" > $output_file

# 时间从 0 开始
time_counter=0
# 执行交易的计数器
transaction_counter=0

# 无限循环，每秒记录一次
while true; do
  # 获取 CPU 和网络带宽占用最高的容器
  docker stats --no-stream --format "{{.Container}},{{.Name}},{{.CPUPerc}},{{.NetIO}}" | sed 's/%//' \
  | awk -F "," '{print $1 "," $2 "," $3 "," $4 "," $5}' \
  | sort -t "," -k3 -nr \
  | head -n 1 | while IFS=',' read -r container_id container_name cpu_usage netio_sent netio_received; do
      # 追加输出到 CSV 文件，格式为: 时间, 容器ID, 容器名称, CPU占用率, 网络发送量, 网络接收量
      echo "$time_counter,$container_id,$container_name,$cpu_usage,$netio_sent,$netio_received"
      echo "$time_counter,$container_id,$container_name,$cpu_usage,$netio_sent,$netio_received" >> $output_file
  done

  # 每2秒执行一次交易，直到执行20次
  if (( time_counter % 2 == 0 && transaction_counter < 20 )); then
    intkey set $(date +%s) $(date +%s) --url http://172.18.20.2:8008
    ((transaction_counter++))
  fi

  # 如果交易执行完20次，等待10秒并结束脚本
  if (( transaction_counter == 20 )); then
    echo "等待 10 秒后结束监控..."
    sleep 10
    break
  fi

  # 每次循环增加时间计数器
  time_counter=$((time_counter + 1))

  # 每隔1秒记录一次
  sleep 1

done