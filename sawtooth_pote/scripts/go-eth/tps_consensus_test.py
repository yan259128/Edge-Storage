from web3 import Web3
import time
import threading
import subprocess
import os
import json
import queue

# 创建队列用于存储线程返回结果
result_queue = queue.Queue()


# 获取 Geth 容器的 IP 地址
def get_container_ip(container_name):
    result = subprocess.run(
        ['docker', 'inspect', '-f', '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}', container_name],
        stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip()


# 从文件中读取账户地址
def read_account_address(node_dir):
    with open(f"{node_dir}/account_address.txt", "r") as f:
        address = f.read().strip()
    return Web3.to_checksum_address(address)


# 从 keystore 文件中解锁账户
def unlock_account_from_keystore(w3, keystore_path, password_path):
    with open(keystore_path, "r") as keyfile:
        encrypted_key = keyfile.read()
    with open(password_path, "r") as pwd_file:
        password = pwd_file.read().strip()
    return w3.eth.account.decrypt(encrypted_key, password)


# 设置节点和账户自动化
def initialize():
    node_count = 4  # 设置节点数量
    nodes = []

    for i in range(1, node_count + 1):
        node_name = f"geth-node{i}"
        node_ip = get_container_ip(node_name)
        account_address = read_account_address(f"node{i}")
        keystore_file = next(file for file in os.listdir(f"node{i}/keystore") if file.startswith("UTC--"))
        keystore_path = f"node{i}/keystore/{keystore_file}"
        password_path = f"node{i}/password.txt"
        http_port = 8545
        http_url = f"http://{node_ip}:{http_port}"
        nodes.append({
            "name": node_name,
            "ip": node_ip,
            "account_address": account_address,
            "keystore_path": keystore_path,
            "password_path": password_path,
            "http_url": http_url
        })

    return nodes


# 检查账户余额
def check_balance(w3, address):
    balance = w3.eth.get_balance(address)
    return w3.from_wei(balance, 'ether')


# 创建交易
def create_transaction(w3, sender, receiver, amount_wei, gas, gas_price, nonce):
    chain_id = w3.eth.chain_id  # 自动获取当前链的chain ID
    tx = {
        'nonce': nonce,
        'to': receiver,
        'value': amount_wei,
        'gas': gas,
        'gasPrice': gas_price,
        'chainId': chain_id  # 加入 chainId
    }
    return tx


# 发送批量交易并记录时间
def send_transactions(w3, sender_address, keystore_path, password_path, receiver_address, num_transactions, amount_eth):
    gas_price = w3.to_wei('20', 'gwei')
    gas = 21000  # 转账交易的标准Gas

    # 解锁账户（从keystore解密）
    private_key = unlock_account_from_keystore(w3, keystore_path, password_path)

    start_time = time.time()

    # 获取初始 nonce
    nonce = w3.eth.get_transaction_count(sender_address)

    for i in range(num_transactions):
        amount_wei = w3.to_wei(amount_eth, 'ether')
        tx = create_transaction(w3, sender_address, receiver_address, amount_wei, gas, gas_price, nonce)
        signed_tx = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        print(f"Transaction {i + 1}/{num_transactions} sent: {tx_hash.hex()}")

        # 增加 nonce
        nonce += 1

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Sent {num_transactions} transactions in {total_time:.2f} seconds")


def send_transaction(w3, sender_address, keystore_path, password_path, receiver_address, amount_eth):
    gas_price = w3.to_wei('20', 'gwei')
    gas = 21000  # 转账交易的标准Gas

    # 解锁账户（从keystore解密）
    private_key = unlock_account_from_keystore(w3, keystore_path, password_path)

    # 获取初始 nonce
    nonce = w3.eth.get_transaction_count(sender_address)

    amount_wei = w3.to_wei(amount_eth, 'ether')
    tx = create_transaction(w3, sender_address, receiver_address, amount_wei, gas, gas_price, nonce)
    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)

    print(f"sent: {tx_hash.hex()}")


total_transactions = 0
generated_blocks = 0
actual_monitoring_time = 0


# 监控区块生成速率（共识速率）
def monitor_block_generation(w3, check_interval=0.1, timeout=200):
    initial_block = w3.eth.block_number
    print(f"Initial block number: {initial_block}")

    start_time = time.time()  # 开始监控时间
    last_block_time = start_time
    global generated_blocks
    global actual_monitoring_time
    global total_transactions

    try:
        while True:
            current_block = w3.eth.block_number
            block = w3.eth.get_block(current_block)
            print(f"Writing time {time.time() - last_block_time}, txns:{len(block.transactions)}")

            # 如果检测到新区块
            if current_block > initial_block + generated_blocks and len(block.transactions) != 0:
                generated_blocks = current_block - initial_block
                last_block_time = time.time()  # 记录最新区块生成时间
                print(f"New block generated: {current_block}, Total blocks generated: {generated_blocks}")

                # 获取该区块的交易数量
                transaction_count = len(block.transactions)
                total_transactions += transaction_count
                print(f"New block {current_block} generated with {transaction_count} transactions.")
                print(f"Total blocks generated: {generated_blocks}, Total transactions: {total_transactions}")

            # 每隔 0.1 秒检查一次
            time.sleep(check_interval)

            # 如果 10 秒内无新区块生成，停止监控
            if time.time() - last_block_time >= timeout:
                print(f"No new block generated in the last {timeout} seconds. Stopping monitoring.")
                break
    except Exception as e:
        print(f"Error occurred: {e}")


    actual_monitoring_time = last_block_time - start_time  # 计算监控时间，减去无效的10秒
    print(f"Total blocks generated: {generated_blocks} in {actual_monitoring_time:.2f} seconds")
    print(f"Total transactions generated: {total_transactions} in {actual_monitoring_time:.2f} seconds")


# 计算TPS和共识速率
def calculate_tps(num_transactions, blocks_generated, duration):
    tps = num_transactions / duration
    consensus_rate = duration / blocks_generated
    print(f"TPS: {tps:.2f} transactions per second")
    print(f"Consensus rate: {consensus_rate:.2f} second per block")


if __name__ == "__main__":
    # 初始化节点配置
    nodes = initialize()

    # 假设我们使用第一个节点作为发送者，第五个节点作为接收者
    sender_node = nodes[0]
    receiver_node = nodes[-1]

    w3_sender = Web3(Web3.HTTPProvider(sender_node['http_url']))
    w3_receiver = Web3(Web3.HTTPProvider(receiver_node['http_url']))

    assert w3_sender.is_connected(), f"Unable to connect to {sender_node['name']}"
    assert w3_receiver.is_connected(), f"Unable to connect to {receiver_node['name']}"

    # 检查账户余额
    print(f"Sender balance: {check_balance(w3_sender, sender_node['account_address'])} ETH")
    print(f"Receiver balance: {check_balance(w3_receiver, receiver_node['account_address'])} ETH")

    # 测试设置
    num_transactions = 5500  # 发送交易数量
    amount_eth = 0.01  # 每笔交易发送0.01 ETH
    timeout = 120  # 监控超时时间

    # 启动区块生成监控线程
    monitor_thread = threading.Thread(target=monitor_block_generation, args=(w3_sender, 0.1, timeout))
    monitor_thread.start()

    # 发送交易并测量时间
    send_transactions(w3_sender, sender_node['account_address'], sender_node['keystore_path'],
                      sender_node['password_path'], receiver_node['account_address'], num_transactions,
                      amount_eth)

    # send_transaction(w3_sender, sender_node['account_address'], sender_node['keystore_path'],
    #                  sender_node['password_path'], receiver_node['account_address'],
    #                  amount_eth)

    # 等待区块监控线程结束
    monitor_thread.join()
    blocks_generated = generated_blocks
    monitoring_duration = actual_monitoring_time
    num_transactions = total_transactions

    # 计算TPS和共识速率
    calculate_tps(num_transactions, blocks_generated, monitoring_duration)
