#!/bin/bash


source ./config.sh

# 生成 intkey 批次文件的脚本

# 从 rest-api-url 文件中读取主机地址
if [ -f "./rest-api-url.txt" ]; then
    REST_API_HOST=$(cat ./rest-api-url.txt)
    URL="http://$REST_API_HOST:8008"  # 自动补充协议和端口
    echo "Using REST API URL: $URL"
else
    echo "Error: rest-api-url file not found!"
    exit 1
fi


# 生成单个批次的函数
generate_batch() {
    local batch_num=$1
    local output_file="${BATCH_PREFIX}${batch_num}.batch"

    echo "Generating batch $batch_num with $TXN_COUNT transactions..."

    # 使用 intkey workload 生成批次
    intkey create_batch --count $TXN_COUNT \
                    --output "$output_file"

    echo "Batch $batch_num generated and saved to $output_file"
}

# 生成所有批次
generate_all_batches() {
    echo "Generating $BATCH_COUNT batches..."

    for ((i=1; i<=BATCH_COUNT; i++)); do
        generate_batch $i
    done

    echo "All batches generated."
}

# 调用函数生成批次
generate_all_batches
