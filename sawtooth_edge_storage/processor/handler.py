"""
定义了交易的元数据,以及交易处理器的逻辑处理。
"""
import hashlib
import logging
from sawtooth_sdk.processor.handler import TransactionHandler
from sawtooth_sdk.processor.exceptions import InvalidTransaction, InternalError
from payload import ESPayload, IS_CREATOR_SIGNER, IS_PARTICIPATOR_SIGNER
from es_state import NAMESPACE, ESState
from validator_state import VALIDATOR_NAMESPACE, ValidatorState

# 配置日志记录
LOGGER = logging.getLogger(__name__)


class ESTransactionHandler(TransactionHandler):
    # ESTransactionHandler 类，继承自 TransactionHandler，用于处理交易。
    @property
    def family_name(self):
        # 定义交易家族名称。
        return 'edge_storage'

    @property
    def family_versions(self):
        # 定义支持的交易家族版本。
        return ['1.0']

    @property
    def namespaces(self):
        # 定义此处理器处理的命名空间。
        return [NAMESPACE, VALIDATOR_NAMESPACE]

    def apply(self, transaction, context):
        # 处理交易的主要逻辑
        header = transaction.header

        # 从交易头部提取出交易的用户公钥
        signer = header.signer_public_key

        # 创建一个 ESPayload 实例
        payload = ESPayload.from_bytes(transaction.payload)

        state = ESState(context)
        validator_state = ValidatorState(context)

        """
        根据负载中的数据处理账本和验证器
        """
        # 创建合约处理
        if payload.action() == 'create':
            LOGGER.info('creating contract %s', payload.contract_hash())
            validator = validator_state.get_validator(payload.data_hash())

            nonce = payload.nonce()
            validate_hash = hashlib.sha256((nonce + payload.data_hash()).encode()).hexdigest()

            # 不为空，说明是转存的，并进行授权判断
            if validator is not None:
                if (validate_hash != validator.validate_hash and (validator.allow_transfer == 'False'
                                                                  or validator.is_transfer == 'True')):
                    LOGGER.error('invalid contract %s, it is an unauthorized transfer or it has been transferred',
                                 payload.contract_hash())
                else:
                    state.set_contract(creator=payload.creator(), participator=payload.participator(),
                                       rule=payload.rule(),
                                       profit=payload.profit(), data_hash=payload.data_hash(),
                                       contract_hash=payload.contract_hash(), is_transfer='True')
                    validator_state.set_transfer(payload.data_hash())
                    LOGGER.info('created contract %s', payload.contract_hash())
                return

            state.set_contract(creator=payload.creator(), participator=payload.participator(), rule=payload.rule(),
                               profit=payload.profit(), data_hash=payload.data_hash(),
                               contract_hash=payload.contract_hash(), is_transfer='False')
            LOGGER.info('created contract %s', payload.contract_hash())
            LOGGER.info('creating validator of contract')
            validator_state.set_validator(payload.data_hash(), validate_hash, payload.creator(), 'False')
            LOGGER.info('Successfully created validator of contract')

        # 展示合约处理，这个目前没用
        elif payload.action() == 'show':
            state.get_contract(contract_hash=payload.contract_hash())

        # 签名处理
        elif payload.action() == 'sign':
            if payload.position() == IS_CREATOR_SIGNER:
                LOGGER.info('creator are signing contract %s', payload.contract_hash())
                state.sign_contract(payload.executor(), payload.contract_hash(), IS_CREATOR_SIGNER)
                LOGGER.info('creator is signed contract %s', payload.contract_hash())
            else:
                LOGGER.info('participator is signing contract %s', payload.contract_hash())
                state.sign_contract(payload.executor(), payload.contract_hash(), IS_PARTICIPATOR_SIGNER)
                LOGGER.info('participator is signed contract %s', payload.contract_hash())

        # 确认处理
        elif payload.action() == 'confirm':
            if payload.position() == IS_CREATOR_SIGNER:
                LOGGER.info('creator is confirming contract %s', payload.contract_hash())
                state.set_execution_result(payload.executor(), payload.contract_hash(), IS_CREATOR_SIGNER)
                LOGGER.info('creator is confirmed contract %s', payload.contract_hash())
            else:
                LOGGER.info('participator is confirming contract %s', payload.contract_hash())
                state.set_execution_result(payload.executor(), payload.contract_hash(), IS_PARTICIPATOR_SIGNER)
                LOGGER.info('participator is confirmed contract %s', payload.contract_hash())

        # 设置允许转存状态处理
        elif payload.action() == 'allow_transfer' or payload.action() == 'not_allow_transfer':
            contract = state.get_contract(payload.contract_hash())
            validator = validator_state.get_validator(contract.data_hash)

            if validator is not None and validator.is_transfer == 'True' and contract.is_transfer == 'True':
                LOGGER.warning('the contract is already a transfer contract and is not allowed to be set to the '
                               'transfer-allowed state.')
                return

            transfer_state = 'True' if payload.action() == 'allow_transfer' else 'False'
            LOGGER.info('creator is allowing transfer contract %s', payload.contract_hash())
            state.set_allow_transfer(payload.contract_hash(), payload.executor(), transfer_state)
            validator_state.set_allow_transfer(contract.data_hash, transfer_state)
            LOGGER.info('creator is allowed transfer contract %s', payload.contract_hash())

        # 删除处理
        elif payload.action() == 'delete':
            LOGGER.info('creator is deleting contract %s', payload.contract_hash())
            state.delete_contract(payload.contract_hash(), payload.executor())
            try:
                contract = state.get_contract(payload.contract_hash())
                validator_state.delete_validator(contract.data_hash, payload.creator())
            except Exception as e:
                LOGGER.warning(e)

            LOGGER.info('creator is deleted contract %s', payload.contract_hash())

        # 如果收到未知的动作，抛出异常。
        else:
            raise InvalidTransaction('Unhandled action: {}'.format(payload.action()))
