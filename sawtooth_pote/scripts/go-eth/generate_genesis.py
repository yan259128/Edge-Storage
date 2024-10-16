import json

# 定义文件路径
genesis_template = {
    "config": {
        "chainId": 1234,
        "homesteadBlock": 0,
        "eip150Block": 0,
        "eip155Block": 0,
        "eip158Block": 0
    },
    "difficulty": "0x20000",
    "gasLimit": "8000000000",
    "alloc": {}
}

NODE_COUNT = 12

# 读取节点的账号地址
node_dirs = []
for i in range(NODE_COUNT):
    node_dirs.append(f"node{i+1}")

for node in node_dirs:
    with open(f"./{node}/account_address.txt", "r") as f:
        address = f.read().strip()

        # 去掉地址前缀 '0x'（确保只有40字符的地址）
        if address.startswith("0x"):
            address = address[2:]

        genesis_template["alloc"][address] = {"balance": "1000000000000000000000"}

# 保存生成的genesis文件
with open("genesis.json", "w") as f:
    json.dump(genesis_template, f, indent=4)

print("genesis.json created successfully.")
