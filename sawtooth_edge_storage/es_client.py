"""
客户端部分。创建交易、发送交易、返回响应以及和交易处理器交互等部分主要在这里实现。
"""

import hashlib
import base64
import json
import time
import random
import requests
import yaml

from sawtooth_edge_storage.es_exception import ESException

from sawtooth_signing import create_context
from sawtooth_signing import CryptoFactory
from sawtooth_signing import ParseError
from sawtooth_signing.secp256k1 import Secp256k1PrivateKey

from sawtooth_sdk.protobuf.transaction_pb2 import TransactionHeader
from sawtooth_sdk.protobuf.transaction_pb2 import Transaction
from sawtooth_sdk.protobuf.batch_pb2 import BatchList
from sawtooth_sdk.protobuf.batch_pb2 import BatchHeader
from sawtooth_sdk.protobuf.batch_pb2 import Batch


# 定义一个函数，用于计算给定数据的SHA-512哈希值
def _sha512(data):
    return hashlib.sha512(data).hexdigest()


FAMILY_NAME = 'edge_storage'
FAMILY_VERSION = '1.0'


# ESClient 类：用于与 Sawtooth 验证器进行交互的客户端
class ESClient:
    # 构造函数：初始化 ESClient 实例
    def __init__(self, base_url, keyfile=None):
        self._base_url = base_url  # Sawtooth REST API 的基础 URL

        # 如果未提供 keyfile，则不创建签名器
        if keyfile is None:
            self._signer = None
            return

        # 从文件读取私钥，并创建签名器
        try:
            with open(keyfile) as fd:
                private_key_str = fd.read().strip()
        except OSError as err:
            # 读取文件失败时抛出异常
            raise ESException(
                'Failed to read private key {}: {}'.format(keyfile, str(err))) from err

        try:
            private_key = Secp256k1PrivateKey.from_hex(private_key_str)
        except ParseError as e:
            # 解析私钥失败时抛出异常
            raise ESException('Unable to load private key: {}'.format(str(e))) from e

        # 创建签名器
        self._signer = CryptoFactory(create_context('secp256k1')).new_signer(private_key)

    # 创建合约
    def create(self, participator, rule, profit, data_hash, wait=None):

        creator = self._signer.get_public_key().as_hex()
        contract_hash = self._get_address(creator + participator + rule + profit + data_hash)
        response = self._send_es_txn("create", creator=creator, wait=wait, participator=participator
                                     , rule=rule, profit=profit, data_hash=data_hash, contract_hash=contract_hash)

        response_data = json.loads(response)
        response_data.update({"contract_hash": contract_hash})
        return json.dumps(response_data)

    # 对合约进行签名
    def sign(self, contract_hash, position, wait=None):
        return self._send_es_txn("sign", contract_hash=contract_hash, position=position, wait=wait)

    # 对执行结果进行签名
    def confirm(self, contract_hash, position, wait=None):
        return self._send_es_txn("confirm", contract_hash=contract_hash, position=position, wait=wait)

    # 删除合约
    def delete(self, contract_hash, wait=None):
        return self._send_es_txn("delete", contract_hash=contract_hash, wait=wait)

    # 设置允许转存状态
    def allow_transfer(self, contract_hash, wait=None):
        return self._send_es_txn("allow_transfer", contract_hash=contract_hash, wait=wait)

    # 设置不允许转存状态
    def not_allow_transfer(self, contract_hash, wait=None):
        return self._send_es_txn("not_allow_transfer", contract_hash=contract_hash, wait=wait)

    # 显示特定合约的方法
    def show(self, contract_hash):
        # 获取特定合约
        address = self._get_address(contract_hash)
        # print(address)

        # 解析并返回合约
        try:
            result = self._send_request("state/{}".format(address), contract_hash=contract_hash)
            return base64.b64decode(yaml.safe_load(result)["data"])
        except BaseException:
            return None

    # 获取交易批处理状态的内部方法
    def _get_status(self, batch_id, wait):
        # 请求批处理状态
        try:
            result = self._send_request('batch_statuses?id={}&wait={}'.format(batch_id, wait))
            return yaml.safe_load(result)['data'][0]['status']
        except BaseException as err:
            raise ESException(err) from err

    # 获取命名空间前缀的内部方法
    def _get_prefix(self):
        return _sha512(FAMILY_NAME.encode('utf-8'))[0:6]

    # 获得验证器的命名空间
    def _get_validator_prefix(self):
        return _sha512((FAMILY_NAME + '_validator').encode('utf-8'))[0:6]

    # 获取地址的内部方法
    def _get_address(self, msg):
        es_prefix = self._get_prefix()
        msg_str = _sha512(msg.encode('utf-8'))[0:64]
        return es_prefix + msg_str

    # 获得验证器的地址，如果msg为空，则返回前缀 + '.*'
    def _get_validator_address(self, msg):
        validator_prefix = self._get_validator_prefix()
        if msg == '':
            return validator_prefix
        msg_str = _sha512(msg.encode('utf-8'))[0:64]
        return validator_prefix + msg_str

    def _send_request(self, suffix, data=None, content_type=None, contract_hash=None):
        """
         发送 HTTP 请求的内部方法
        :param suffix:  请求的URL后缀。如果基础URL是 http://127.0.0.1:8008, suffix是 state/xxx，那么完整的URL是 http://127.0.0.1/state/xxx。
        :param data:    用于POST请求的结构体。如果参数被提供则执行一个POST请求，反之执行一个GET请求。
        :param content_type: 指定HTTP请求的Content-Type头部。
        :param contract_hash: 当前合约的hash
        :return: 返回请求结果
        """

        # 构建请求 URL
        url = "{}/{}".format(self._base_url, suffix) if self._base_url.startswith("http://") else "http://{}/{}".format(
            self._base_url, suffix)

        # print('url: {}'.format(url))
        # 设置认证和内容类型头
        headers = {}
        if content_type is not None:
            headers['Content-Type'] = content_type

        # 发送请求并处理响应
        try:
            result = requests.post(url, headers=headers, data=data) if data is not None else requests.get(
                url, headers=headers)
            if result.status_code == 404:
                raise ESException("No such contract contract_hash: {}".format(contract_hash))
            if not result.ok:
                raise ESException("Error {}: {}".format(result.status_code, result.reason))
        except requests.ConnectionError as err:
            raise ESException('Failed to connect to {}: {}'.format(url, str(err))) from err
        except BaseException as err:
            raise ESException(err) from err

        return result.text

    def _send_es_txn(self, action, creator="", participator="", rule="", profit="", data_hash="", contract_hash=None,
                     position='0', wait=None):
        """
        发送交易
        :param action: 操作的类型：'create' -- 创建合约， 'show' -- 显示指定合约， 'sign' -- 对指定合约进行签名，  'confirm' -- 对合约执行结果进行确认
            'delete -- 删除合约， ‘allow_transfer' -- 设置允许转存的状态, 'not_allow_transfer' -- 设置不被允许转存的状态；
        :param creator: 合约创建者的公钥；
        :param participator: 合约参与者公钥；
        :param rule: 合约规则；
        :param profit: 利益分配；
        :param data_hash: 数据hash；
        :param contract_hash: 当前合约的hash；
        :param position: 签名位置
        :param wait: 等待时间；
        :return: 返回处理信息。
        """

        # 序列化交易载荷
        nonce = hex(random.randint(0, 2 ** 64))
        executor = ""

        if (action == 'sign' or action == 'confirm' or action == 'delete'
                or action == 'allow_transfer' or action == 'not_allow_transfer'):
            executor = self._signer.get_public_key().as_hex()

        payload = (",".join([action, creator, participator, rule, profit, data_hash, contract_hash, executor,
                             position, nonce]).encode())

        # 构建交易头部
        address = self._get_address(contract_hash)
        validator_address = self._get_validator_address(data_hash)
        header = TransactionHeader(
            signer_public_key=self._signer.get_public_key().as_hex(),
            family_name=FAMILY_NAME,
            family_version=FAMILY_VERSION,
            inputs=[address, validator_address],
            outputs=[address, validator_address],
            dependencies=[],
            payload_sha512=_sha512(payload),
            batcher_public_key=self._signer.get_public_key().as_hex(),
            nonce=nonce
        ).SerializeToString()

        # 签名交易头部并创建交易
        signature = self._signer.sign(header)
        transaction = Transaction(header=header, payload=payload, header_signature=signature)

        # 创建批处理列表
        batch_list = self._create_batch_list([transaction])
        batch_id = batch_list.batches[0].header_signature

        # 发送批处理列表，并等待处理结果（如果指定了等待时间）
        if wait and wait > 0:
            response = self._send_request("batches", batch_list.SerializeToString(), 'application/octet-stream')
            wait_time = 0
            start_time = time.time()
            while wait_time < wait:
                status = self._get_status(batch_id, wait - int(wait_time))
                wait_time = time.time() - start_time
                if status != 'PENDING':
                    return response
            return response

        return self._send_request("batches", batch_list.SerializeToString(), 'application/octet-stream')

    # 创建批处理列表的内部方法
    def _create_batch_list(self, transactions):
        # 构建批处理头部
        transaction_signatures = [t.header_signature for t in transactions]
        header = BatchHeader(signer_public_key=self._signer.get_public_key().as_hex(),
                             transaction_ids=transaction_signatures).SerializeToString()

        # 签名批处理头部并创建批处理
        signature = self._signer.sign(header)
        batch = Batch(header=header, transactions=transactions, header_signature=signature)
        return BatchList(batches=[batch])
