"""
定义了对交易验证器的实际操作，类似于智能合约的地方。
"""

import hashlib

from sawtooth_sdk.processor.exceptions import InternalError
from sawtooth_sdk.processor.exceptions import InvalidTransaction

# 定义了一个常量，表示命名空间。这个命名空间是通过对字符串'edge_storage'进行SHA-512哈希运算然后取前6个字符得到的。
VALIDATOR_NAMESPACE = hashlib.sha512('edge_storage_validator'.encode("utf-8")).hexdigest()[0:6]


def _make_address(msg):
    """
    根据信息生成一个唯一的区块链地址。这是通过将msg进行SHA-512哈希运算然后取前64个字符，并与命名空间拼接得到的。
    :param msg: 用来生成地址的信息。(str)
    :return: hash
    """
    return VALIDATOR_NAMESPACE + \
        hashlib.sha512(msg.encode('utf-8')).hexdigest()[:64]


# 验证类，存储合约的验证信息，防止未授权转存。
class ContractValidator:
    def __init__(self, data_hash, validate_hash, creator, allow_transfer, is_transfer):
        self.data_hash = data_hash
        self.validate_hash = validate_hash
        self.creator = creator
        self.allow_transfer = allow_transfer
        self.is_transfer = is_transfer


# 管理验证器状态的类
class ValidatorState:
    TIMEOUT = 3

    def __init__(self, context):
        self._context = context

    def set_validator(self, data_hash, validate_hash, creator, allow_transfer):
        """
        生成一个验证器
        :param allow_transfer: 是否允许转存的标识 ('True' | 'False')
        :param data_hash:    当前数据的hash (str)
        :param validate_hash: 验证用hash (str)
        :param creator: 创建者公钥
        """
        validator = ContractValidator(data_hash, validate_hash, creator, allow_transfer, 'False')
        self._store_validators(data_hash, validator)

    def get_validator(self, data_hash):
        """
        加载验证信息
        :param data_hash: 当前数据的hash (str)
        :return: 验证器实例 (ContractValidator), 不存在则返回 None
        """
        return self._load_validators(data_hash)

    def delete_validator(self, data_hash, remover):
        """
        删除验证器，以便下次
        :param remover: 删除人私钥
        :param data_hash: 当前数据的hash (str)
        """
        validator = self.get_validator(data_hash)
        if validator.creator != remover:
            raise InvalidTransaction("Only the creator can delete the validator")

        return self._delete_validator(data_hash)

    def set_transfer(self, data_hash):
        """
        设置保存的当前验证信息的合约已经被转存过
        :param data_hash: 当前数据的hash (str)
        """
        validator = self.get_validator(data_hash)
        validator.is_transfer = 'True'
        self._store_validators(data_hash, validator)

    def set_allow_transfer(self, data_hash, state):
        """
        设置允许转存
        :param state: 是否允许转存的状态 ('True' | 'False')
        :param data_hash: 当前数据的hash (str)
        """
        validator = self.get_validator(data_hash)
        validator.allow_transfer = state
        self._store_validators(data_hash, validator)

    def _load_validators(self, data_hash):
        """
        从区块链中加载验证信息
        :param data_hash: 当前数据的hash (str)
        :return: 验证器实例 (ContractValidator), 不存在则返回 None
        """

        address = _make_address(data_hash)
        try:
            state_entries = self._context.get_state([address], timeout=self.TIMEOUT)
        except Exception:
            return None

        if state_entries:
            return self._deserialize(state_entries[0].data)
        else:
            return None

    def _store_validators(self, data_hash, validator):
        """
        存储验证器状态到区块链中
        :param data_hash: 当前数据的hash (str)
        :param validator: 验证器实例 (ContractValidator)
        """

        address = _make_address(data_hash)
        data = self._serialize(validator)
        self._context.set_state({address: data}, timeout=self.TIMEOUT)

    def _delete_validator(self, data_hash):
        """
        删除验证器
        :param data_hash: 当前数据的hash (str)
        """

        address = _make_address(data_hash)

        self._context.delete_state(
            [address],
            timeout=self.TIMEOUT)

    def _deserialize(self, data):
        """
        反序列化
        :param data: 序列化后的验证数据 (str)
        :return: 验证器实例 (Contract)
        """

        try:
            data_hash, validate_hash, creator, allow_transfer, is_transfer = data.decode().split(",")
            validator = ContractValidator(data_hash, validate_hash, creator, allow_transfer, is_transfer)
        except ValueError as e:
            raise InternalError("Failed to deserialize game data") from e

        return validator

    def _serialize(self, validator):
        """
        序列化
        :param validator: 验证器实例 (ContractValidator)
        :return: 序列化后的合约数据 (str)
        """

        data = ",".join([validator.data_hash, validator.validate_hash, validator.creator, validator.allow_transfer,
                         validator.is_transfer])

        return data.encode()
