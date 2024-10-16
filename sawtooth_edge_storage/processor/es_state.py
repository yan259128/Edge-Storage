"""
定义了对账本和数据的实际操作，类似于智能合约的地方。
"""

import hashlib

from sawtooth_sdk.processor.exceptions import InternalError
from sawtooth_sdk.processor.exceptions import InvalidTransaction
from payload import IS_PARTICIPATOR_SIGNER, IS_CREATOR_SIGNER

# 定义了一个常量，表示命名空间。这个命名空间是通过对字符串'edge_storage'进行SHA-512哈希运算然后取前6个字符得到的。
NAMESPACE = hashlib.sha512('edge_storage'.encode("utf-8")).hexdigest()[0:6]


def _make_address(msg):
    """
    根据信息生成一个唯一的区块链地址。这是通过将msg进行SHA-512哈希运算然后取前64个字符，并与命名空间拼接得到的。
    :param msg: 用来生成地址的信息。(str)
    :return: hash
    """
    return NAMESPACE + \
        hashlib.sha512(msg.encode('utf-8')).hexdigest()[:64]


# 合约类，用于表示一个合约的状态
class Contract:
    def __init__(self, creator, participator, rule, profit, data_hash, is_transfer, is_signed_by_creator="",
                 is_signed_by_participator="", execution_result='False', confirmation_result='False',
                 allow_transfer='False'):
        self.creator = creator  # 创建者公钥
        self.participator = participator  # 参与者公钥
        self.rule = rule  # 交易规则
        self.profit = profit  # 利益分配
        self.data_hash = data_hash  # 数据Hash
        self.is_signed_by_creator = is_signed_by_creator  # 创建者签名状态
        self.is_signed_by_participator = is_signed_by_participator  # 参与者签名状态
        self.execution_result = execution_result  # 参与者执行结果
        self.confirmation_result = confirmation_result  # 创建者对执行结果的确认
        self.allow_transfer = allow_transfer  # 表示是否允许被转存, 'False' | 'True'
        self.is_transfer = is_transfer  # 表示当前合约是否是转存合约，'False' | 'True'


# 管理合约状态的类
class ESState:
    TIMEOUT = 3  # 操作的超时时间

    def __init__(self, context):
        self._context = context

    def set_contract(self, creator, participator, rule, profit, data_hash, contract_hash, is_transfer):
        # 创建合约
        """
        创建合约
        :param is_transfer: 判断当前合约是否是转存的标识 ('False' | 'True')
        :param creator: 创建者公钥 (str)
        :param participator: 参与者公钥 (str)
        :param rule: 交易规则 (str)
        :param profit: 利益分配 (str)
        :param data_hash: 数据Hash (str)
        :param contract_hash: 当前合约的hash（str）
        """

        contract = Contract(creator, participator, rule, profit, data_hash, is_transfer)
        self._store_contracts(contract_hash, contract)

    def get_contract(self, contract_hash):
        """
        获得指定合约
        :param contract_hash: 当前合约的hash (str)
        :return: 合约实例 (Contract)
        """
        return self._load_contracts(contract_hash)

    def sign_contract(self, executor, contract_hash, position):
        """
        对合约进行签名
        :param executor: 执行人账号公钥 (str)
        :param contract_hash: 当前合约的hash (str)
        :param position: 签名位置（str）, 创建者的位置是 0 ， 参与者的位置是 1
        :return:
        """
        contract = self.get_contract(contract_hash)

        if position == IS_CREATOR_SIGNER:
            if executor != contract.creator:
                raise InvalidTransaction("Invalid signature. Only the contract creator can sign at this location")
            else:
                contract.is_signed_by_creator = executor

        if position == IS_PARTICIPATOR_SIGNER:
            if executor != contract.participator:
                raise InvalidTransaction("Invalid signature. Only the contract participator can sign at this location")
            else:
                contract.is_signed_by_participator = executor

        self._store_contracts(contract_hash, contract)

    def set_execution_result(self, executor, contract_hash, position):
        """
        设置执行结构
        :param executor: 执行者账公钥 (str)
        :param contract_hash: 当前合约的hash (str)
        :param position: 确认位置（str）, 创建者的位置是 0 ， 参与者的位置是 1
        """
        contract = self.get_contract(contract_hash)

        if position == IS_CREATOR_SIGNER:
            if executor != contract.creator:
                raise InvalidTransaction("Invalid confirmation. Only the contract creator can confirm at this location")
            else:
                contract.confirmation_result = "True"

        if position == IS_PARTICIPATOR_SIGNER:
            if executor != contract.participator:
                raise InvalidTransaction("Invalid confirmation. Only the contract participator can confirm at this "
                                         "location")
            else:
                contract.execution_result = "True"

        self._store_contracts(contract_hash, contract)

    def delete_contract(self, contract_hash, remover):
        """
        删除验证器，以便下次
        :param remover: 删除者公钥 (str)
        :param contract_hash: 当前合约的hash (str)
        """
        contract = self.get_contract(contract_hash)
        if contract.creator != remover:
            raise InvalidTransaction("Only the contract creator be delete contract")
        return self._delete_contract(contract_hash)

    def set_allow_transfer(self, contract_hash, executor, state):
        """
        设置允许转存
        :param state: 是否允许转存的状态 ('True' | 'False')
        :param executor:  执行者公钥 (str)
        :param contract_hash: 该合约的hash (str)
        """
        contract = self.get_contract(contract_hash)
        if executor == contract.creator:
            contract = self.get_contract(contract_hash)
            contract.allow_transfer = state
        else:
            raise InvalidTransaction("Only the creator is allowed transfer contract")

        self._store_contracts(contract_hash, contract)

    def _load_contracts(self, contract_hash):
        """
        从区块链中加载合约状态
        :param contract_hash: 当前合约的hash (str)
        :return: 合约实例 (Contract)
        """

        address = _make_address(contract_hash)
        state_entries = self._context.get_state([address], timeout=self.TIMEOUT)
        if state_entries:
            return self._deserialize(state_entries[0].data)
        else:
            return None

    def _store_contracts(self, contract_hash, contract):
        """
        存储合约状态到区块链中
        :param contract_hash: 当前合约的hash (str)
        :param contract: 合约实例 (Contract)
        """

        address = _make_address(contract_hash)
        data = self._serialize(contract)
        self._context.set_state({address: data}, timeout=self.TIMEOUT)

    def _delete_contract(self, contract_hash):
        """
        删除验证器
        :param contract_hash: 当前数据的hash (str)
        """
        address = _make_address(contract_hash)

        self._context.delete_state(
            [address],
            timeout=self.TIMEOUT)

    def _deserialize(self, data):
        """
        反序列化
        :param data: 序列化后的合约数据 (str)
        :return: 合约实例 (Contract)
        """

        try:
            creator, participator, rule, profit, data_hash, is_transfer, is_signed_by_creator, \
                is_signed_by_participator, execution_result, confirmation_result, allow_transfer = (data.decode()
                                                                                                    .split(","))

            contract = Contract(creator=creator, participator=participator, rule=rule, profit=profit,
                                data_hash=data_hash, is_signed_by_creator=is_signed_by_creator,
                                is_signed_by_participator=is_signed_by_participator, execution_result=execution_result,
                                confirmation_result=confirmation_result, allow_transfer=allow_transfer,
                                is_transfer=is_transfer)

        except ValueError as e:
            raise InternalError("Failed to deserialize game data") from e

        return contract

    def _serialize(self, contract):
        """
        序列化
        :param contract: 合约实例 (ContractValidator)
        :return: 序列化后的合约数据 (str)
        """

        data = ",".join([contract.creator, contract.participator, contract.rule, contract.profit,
                         contract.data_hash, contract.is_transfer, contract.is_signed_by_creator,
                         contract.is_signed_by_participator, contract.execution_result,
                         contract.confirmation_result, contract.allow_transfer])

        return data.encode()
