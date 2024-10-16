"""
交易的接口演示，这里只演示了创建合约和显示合约内容的接口。其他接口参照这两个调用就行
"""
import json
import time

from sawtooth_edge_storage.es_client import ESClient
import getpass
import os

PARTICIPATOR = '0x000001'   # '交易参与者'
RULE = 'rule'               # '合约规则'
PROFIT = 'profit'           # '利益分配'
DATA_HASH = 'hash'          # '数据哈希' 由于处理交易的逻辑问题，所以请保证每次提交交易的数据哈希都不一样。否则会报错。（下面的代码的接口加入了时间戳保证不一样。）
TIME = time.time()          # 时间戳
WAIT = None  # 等待时长

# 获取rest api地址
with open("rest-api-url.txt", "r") as file:
    address = file.read().strip()
    rest_api_url = f"http://{address}:8008"

# 获取节点用户名
with open("username.txt", "r") as file:
    username = file.read().strip()


# 获取密钥文件
def _get_keyfile(username):
    home = os.path.expanduser("~")
    key_dir = os.path.join(home, ".sawtooth", "keys")
    return '{}/{}.priv'.format(key_dir, username)


# 创建合约
def create_contract():
    client = ESClient(rest_api_url, keyfile=_get_keyfile(username))

    # 添加时间戳，避免每次提交的合约内容相同
    response = client.create(PARTICIPATOR, RULE, PROFIT, DATA_HASH + f'{TIME}', WAIT)
    print("Response: {}".format(response))
    return json.loads(response)['contract_hash']


# 显示合约
def show_contract(contract_hash):
    client = ESClient(rest_api_url, keyfile=_get_keyfile(username))

    response = client.show(contract_hash)

    if response is not None:
        creator, participator, rule, profit, data_hash, is_transfer, is_signed_by_creator, \
            is_signed_by_participator, execution_result, confirmation_result, allow_transfer = (response.decode()
                                                                                                .split(","))

        print("CREATOR                 : ", creator)
        print("PARTICIPATOR            : ", participator)
        print("RULE                    : ", rule[:10])
        print("PROFIT                  : ", profit[:10])
        print("DATA HASH               : ", data_hash)
        print("SIGNED BY CREATOR       : ", is_signed_by_creator)
        print("SIGNED BY PARTICIPATOR  : ", is_signed_by_participator)
        print("EXECUTION RESULT        : ", execution_result)
        print("CONFIRMATION RESULT     : ", confirmation_result)
        print("ALLOW TRANSFER          : ", allow_transfer)
        print("IS TRANSFER             : ", is_transfer)
        return True

    else:
        return False


"""
以下是执行部分（显示合约的功能不会向区块链提交交易）
"""
print("Creating contract...")
contract_hash = create_contract()  # 提交交易（创建合约的交易）
print("Waiting for consensus...")

# 一直循环，直到能够查询到提交的合约（完成了共识，上链了）。
# 注意：提交时请保证data_hash不一样，否则会处理器会报错。如果合约内容完全一样，那么报错后依然可以查询到，不过是上一次提交的。
print(contract_hash)
while show_contract(contract_hash) is False:
    time.sleep(1)
