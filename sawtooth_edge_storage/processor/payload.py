"""
定义了客户端传来的数据结构，以及定义如何被解析和验证。
"""

from sawtooth_sdk.processor.exceptions import InvalidTransaction

IS_CREATOR_SIGNER = '0'
IS_PARTICIPATOR_SIGNER = '1'


class ESPayload:
    def __init__(self, payload):
        try:
            # 负载是一个以逗号分隔的utf-8编码的字符串。
            (action, creator, participator, rule, profit, data_hash,
             contract_hash, executor, position, nonce) = payload.decode().split(",")

        except ValueError as e:
            # 如果负载无法正确解析，抛出 InvalidTransaction 异常。
            raise InvalidTransaction("Invalid payload serialization") from e

        if not creator and action == "create":
            # 如果创建者为空，抛出 InvalidTransaction 异常。
            raise InvalidTransaction("Creator is required")

        if not participator and action == "create":
            # 如果参与者为空，抛出 InvalidTransaction 异常。
            raise InvalidTransaction("Participator is required")

        if action == 'sign' and position != IS_CREATOR_SIGNER and position != IS_PARTICIPATOR_SIGNER:
            # 签名的位置错误
            raise InvalidTransaction("Wrong signature position:"+position+".It can only be 0 or 1")

        if action == 'confirm' and position != IS_CREATOR_SIGNER and position != IS_PARTICIPATOR_SIGNER:
            # 执行的位置错误
            raise InvalidTransaction("Wrong confirmation position:"+position+".It can only be 0 or 1")

        if contract_hash == "":
            raise InvalidTransaction("contract_hash is required")

        self._action = action
        self._creator = creator
        self._participator = participator
        self._rule = rule
        self._profit = profit
        self._data_hash = data_hash
        self._contract_hash = contract_hash
        self._executor = executor
        self._position = position
        self._nonce = nonce

    @staticmethod
    def from_bytes(payload):
        # 静态方法，从字节数据创建一个 ESPayload 实例。
        return ESPayload(payload=payload)

    def creator(self):
        return self._creator

    def participator(self):
        return self._participator

    def rule(self):
        return self._rule

    def profit(self):
        return self._profit

    def data_hash(self):
        return self._data_hash

    def action(self):
        return self._action

    def contract_hash(self):
        return self._contract_hash

    def executor(self):
        return self._executor

    def position(self):
        return self._position

    def nonce(self):
        return self._nonce
