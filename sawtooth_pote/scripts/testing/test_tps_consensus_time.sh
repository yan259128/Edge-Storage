#!/bin/bash

source ./config.sh

# 从 rest-api-url 文件中读取主机地址
if [ -f "./rest-api-url.txt" ]; then
    REST_API_HOST=$(cat ./rest-api-url.txt)
    URL="http://$REST_API_HOST:8008"  # 自动补充协议和端口
    echo "Using REST API URL: $URL"
else
    echo "Error: rest-api-url file not found!"
    exit 1
fi



# 获取最新区块号
get_latest_block_num() {
    latest_block_num=$(curl -s "$URL/blocks?limit=1" | jq -r '.data[0].header.block_num')
    echo $latest_block_num
}

# 获取从指定区块开始的所有交易总数
get_total_txns_from_block() {
    local start_block_num=$1

    block_txns=$(sawtooth block list --url "$URL" -F json | jq "[.[] | select(.num | tonumber > $start_block_num)] | map(.txns) | add")


    echo $block_txns
}




# 提交批次并测量 TPS 和带宽
submit_batches_and_measure_tps_and_consensus_time() {
    echo "Submitting $BATCH_COUNT batches and measuring TPS..."


    start_block_num=$(get_latest_block_num)
    echo "start block num: $start_block_num"

    # 并行提交所有批次
    for ((i=1; i<=BATCH_COUNT; i++)); do
        curl -X POST -H "Content-Type: application/octet-stream" \
             --data-binary @$BATCH_PREFIX$i.batch $URL/batches &
    done

    # 等待所有并行提交任务完成
    wait

    # 记录初始时间
    start_time=$(date +%s%N)

    # 记录上一轮的区块号，用于判断是否处理完毕
    previous_block_num=$start_block_num
    stable_count=0  # 稳定检查计数器
    stable_time=0  # 稳定等待的累计时间


    while true; do
        sleep $CHECK_INTERVAL  # 检测交易处理进度的间隔，可以调节

        # 获取最新区块号
        current_block_num=$(get_latest_block_num)

        # 如果最新区块号没有变化，增加稳定计数并累计等待时间
        if [[ "$current_block_num" == "$previous_block_num" ]]; then
            stable_count=$((stable_count + 1))
            stable_time=$(echo "$stable_time + $CHECK_INTERVAL" | bc)
        else
            stable_count=0  # 如果区块号有变化，重置计数器
            stable_time=0
        fi

        # 当达到稳定次数时，退出循环
        if [[ $stable_count -ge $STABLE_CHECKS ]]; then
            break
        fi

        if [[ $stable_count -ge 1 ]]; then
          echo "Waited $stable_count count for stable block number"
        fi

        # 更新上一轮区块号
        previous_block_num=$current_block_num
    done

    # 记录结束时间
    END_TIME=$(date +%s%N)
    echo "end time: $END_TIME"

    total_expected_txns=$(get_total_txns_from_block $start_block_num)
    echo "total txns: $total_expected_txns"

    TOTAL_TIME=$(echo "scale=6; ($END_TIME - $start_time) / 1000000000 - $stable_time" | bc)

    # 计算最终的总 TPS 和 Consensus Time
    FINAL_TPS=$(echo "scale=2; $total_expected_txns / $TOTAL_TIME" | bc -l)

    echo "Consensus time per block: $(echo "scale=6; $TOTAL_TIME / ($current_block_num - $start_block_num)" | bc) seconds"
    echo "Final TPS: $FINAL_TPS (Total transactions: $total_expected_txns, Total time: $TOTAL_TIME seconds)"
}



# 调用函数进行测试
submit_batches_and_measure_tps_and_consensus_time
