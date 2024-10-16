#!/bin/bash

BATCH_COUNT=50  # 批次数
BATCH_PREFIX="intkey_batch_"  # 批次文件前缀
TXN_COUNT=20  # 随机数key

# 输出文件名
output_file="cpu_bandwidth_usage.csv"

STABLE_CHECKS=200  # 稳定次数
CHECK_INTERVAL=0.1
