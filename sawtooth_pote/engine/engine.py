"""
共识引擎。负责共识机制的实现以及与sawtooth网络的交互。
"""

import json
import logging
import queue
import base64
# import numpy as np
import datetime
import os
import sys
import random

import sawtooth_pote.consensus.vrf as vrf
from sawtooth_sdk.consensus.engine import Engine
from sawtooth_sdk.protobuf.validator_pb2 import Message
from sawtooth_sdk.consensus import exceptions
from sawtooth_pote.consensus.validator_registry_pb2 import ValidatorInfo
from sawtooth_pote.engine.oracle import PoTEBlock, _load_identity_signer, _load_identify_key
from sawtooth_pote.engine.pending import PendingForks

LOGGER = logging.getLogger(__name__)


class PoTEEngine(Engine):
    def __init__(self, path_config, component_endpoint):
        self._path_config = path_config
        self._component_endpoint = component_endpoint
        self._service = None

        # State Variable
        self._exit = False
        self._published = False
        self._building = False
        self._committing = False

        self._validating_blocks = set()     # 存储验证中的区块
        self._pending_forks_to_resolve = PendingForks()     # 存储分叉的数据结构
        self._validators_info_dict = {}     # 存储验证者的字典
        self._wait_time = None      # 接受其他节点信息的等待时间

    def name(self):
        return "pote"

    def version(self):
        return "0.1"

    def additional_protocols(self):
        return [('pote', '0.1')]

    def stop(self):
        self._exit = True

    def start(self, updates, service, startup_state):

        print('engine start')
        self._service = service
        LOGGER.info(msg="PoTE Consensus Engine starting...")

        local_id = startup_state.local_peer_info.peer_id
        current_block_id = startup_state.chain_head.block_id
        settings = self._service.get_settings(current_block_id, ["sawtooth.consensus.pote.example"])

        LOGGER.info(msg="Local ID: {}".format(local_id))
        LOGGER.info(msg="Current Block ID: {}".format(current_block_id))
        LOGGER.info(msg="Example setting: {}".format(settings["sawtooth.consensus.pote.example"]))

        handlers = {
            Message.CONSENSUS_NOTIFY_BLOCK_NEW: self._handle_new_block,
            Message.CONSENSUS_NOTIFY_BLOCK_VALID: self._handle_valid_block,
            Message.CONSENSUS_NOTIFY_BLOCK_INVALID: self._handle_invalid_block,
            Message.CONSENSUS_NOTIFY_BLOCK_COMMIT: self._handle_committed_block,
            Message.CONSENSUS_NOTIFY_PEER_CONNECTED: self._handle_peer_connected,
            Message.CONSENSUS_NOTIFY_PEER_DISCONNECTED: self._handle_peer_disconnected,
            Message.CONSENSUS_NOTIFY_PEER_MESSAGE: self._handle_peer_msgs,
        }

        if startup_state.chain_head.previous_id == b'\x00\x00\x00\x00\x00\x00\x00\x00':
            LOGGER.info("Genesis block detected")

        else:
            LOGGER.info("Non genesis block detected")

        while True:
            try:
                try:
                    type_tag, data = updates.get(timeout=0.1)
                except queue.Empty:
                    pass
                else:
                    LOGGER.debug('Received message: %s',
                                 Message.MessageType.Name(type_tag))
                    try:
                        handle_message = handlers[type_tag]
                    except KeyError:
                        LOGGER.error('Unknown type tag: %s',
                                     Message.MessageType.Name(type_tag))
                    else:
                        handle_message(data)

                if self._exit:
                    break

                self._try_to_publish()

            # pylint: disable=broad-except
            except Exception:
                LOGGER.exception("Unhandled exception in message loop")

    # 获得当前链头
    def _get_chain_head(self):
        return PoTEBlock(self._service.get_chain_head())

    def _initialize_block(self):

        chain_head = self._get_chain_head()

        # 加载Signer对象和获得私钥
        signer = _load_identity_signer(self._path_config.key_dir, 'validator')
        sk_bytes = _load_identify_key(self._path_config.key_dir, 'validator')

        # 通过 vrf 生成 proof
        p_status, proof = vrf.ecvrf_prove(sk_bytes[:32], chain_head.block_id)

        # 生成随机数
        b_status, random_num = vrf.ecvrf_proof_to_hash(proof)
        if p_status != "VALID" or b_status != "VALID":
            LOGGER.error("Failed to generate VRF proof")
            return False

        validator_id = signer.get_public_key().as_hex()
        expectation = random.random()  # 生成 0-1之间的随机数，测试用

        LOGGER.info("Generate VRF random number %s, "
                    "expectation: %s", random_num.hex(), expectation)

        self._wait_time = datetime.datetime.now() + datetime.timedelta(seconds=3)
        validator = ValidatorInfo(id=validator_id, proof=proof, previous_id=chain_head.block_id,
                                  proof_data=chain_head.block_id, expectation=expectation,
                                  name='validator', public_key=vrf.get_public_key(sk_bytes),
                                  random=random_num)

        self._service.initialize_block(previous_id=chain_head.block_id)
        LOGGER.info("Initialized block")

        # 广播到其他节点
        self._broadcast(validator)

        validators = self._validators_info_dict.get(chain_head.block_id, None)
        if validators is None:
            self._validators_info_dict[chain_head.block_id] = [validator]
        elif validator not in validators:
            self._validators_info_dict[chain_head.block_id].append(validator)

        return True

    # 进行最终发布区块前的检查
    def _check_publish_block(self):
        # 检查是否等待完成特定时间，以收集足够的节点发来的信息
        now_time = datetime.datetime.now()
        return now_time >= self._wait_time

    def _finalize_block(self):
        # summary = self._summarize_block()

        if not self._validators_info_dict:
            return None

        # if summary is None:
        #     LOGGER.debug("Block not ready to be summarized")
        #     return None
        LOGGER.info("Finalizing block")
        # 按照期望排序并选出前2/3中随机数最小的验证者作为记账节点
        validators_of_min_random = []
        for _, validators in self._validators_info_dict.items():
            validators = sorted(validators, key=lambda x: x.expectation, reverse=True)
            validator_min_random = validators[0]
            if len(validators) > 1:
                validators = validators[:int(len(validators) * (2 / 3))]
                for info in validators[1:]:
                    if info.random < validator_min_random.random:
                        validator_min_random = info
            validators_of_min_random.append(validator_min_random)

        validator_id = (_load_identity_signer(self._path_config.key_dir, 'validator')
                        .get_public_key().as_hex())
        chain_head = self._get_chain_head()

        # 检查是否被选为记账节点以及判断是否该验证者参与选举的区块
        selected = None
        for validator in validators_of_min_random:
            if validator_id == validator.id and chain_head.block_id == validator.previous_id:
                selected = validator
                del self._validators_info_dict[validator.previous_id]
                break
            else:
                del self._validators_info_dict[validator.previous_id]

        if selected is None:
            LOGGER.debug('This is not selected')
            return None

        # 调用 service 中的finalize_block方法来创建区块，并传输共识信息
        try:
            consensus = json.dumps({
                'public_key': base64.b64encode(selected.public_key).decode('utf-8'),
                'proof': base64.b64encode(selected.proof).decode('utf-8'),
                'proof_data': base64.b64encode(selected.proof_data).decode('utf-8'),
                'random': base64.b64encode(selected.random).decode('utf-8')}
            ).encode()
            block_id = self._service.finalize_block(consensus)
            LOGGER.info("The vrf public key of the successfully elected validator: %s",
                        selected.public_key.hex())
            return block_id
        except exceptions.BlockNotReady:
            LOGGER.debug("Block not ready to be finalized")
            return None
        except exceptions.InvalidState:
            LOGGER.debug("block cannot be finalized")
            return None

    def _cancel_block(self):
        try:
            self._service.cancel_block()
        except exceptions.InvalidState:
            pass

    # 总结当前区块的状态。
    def _summarize_block(self):
        try:
            return self._service.summarize_block()
        except exceptions.InvalidState as err:
            LOGGER.warning(err)
            return None
        except exceptions.BlockNotReady:
            return None

    # 尝试发布区块
    def _try_to_publish(self):
        if self._published or self._validating_blocks:
            return

        if not self._building:
            if self._initialize_block():
                self._building = True

        if self._building:
            if self._check_publish_block():
                block_id = self._finalize_block()
                if block_id:
                    LOGGER.info('Published block: %s', block_id.hex())
                    self._published = True
                    self._building = False
                    self._wait_time = None
                else:
                    self._cancel_block()
                    self._building = False
                    self._wait_time = None

    def _check_block(self, block_id):
        self._service.check_blocks([block_id])

    def _get_block(self, block_id):
        return PoTEBlock(self._service.get_blocks([block_id])[block_id])

    def _fail_block(self, block_id):
        self._service.fail_block(block_id)

    def _process_pending_forks(self):
        pass

    def _resolve_fork(self, block):
        pass

    def _switch_forks(self, current_head, new_head):
        pass

    def _commit_block(self, block_id):
        self._service.commit_block(block_id)

    def _handle_new_block(self, block):
        block = PoTEBlock(block)
        LOGGER.info("HANDLING NEW BLOCK: %s", block)

        self._check_block(block.block_id)
        self._validating_blocks.add(block.block_id)

    def _handle_valid_block(self, block_id):
        self._validating_blocks.discard(block_id)
        block = self._get_block(block_id)
        LOGGER.info("HANDLING VALID BLOCK %s", block)

        if block is not None:
            # 提取共识信息
            consensus_info = None
            try:
                consensus_info = json.loads(block.header.consensus.decode())
            except (json.decoder.JSONDecodeError, KeyError):
                pass

            if consensus_info is None:
                LOGGER.error('Invalid consensus')
                return

            # 解码
            public_key = base64.b64decode(consensus_info['public_key'])
            proof = base64.b64decode(consensus_info['proof'])
            proof_data = base64.b64decode(consensus_info['proof_data'])
            random = base64.b64decode(consensus_info['random'])

            # vrf 验证
            result, random_bytes = vrf.ecvrf_verify(public_key, proof, proof_data)
            if result == 'VALID' and random_bytes == random:
                LOGGER.info('Passed consensus check: %s', block.block_id.hex())
                # self._pending_forks_to_resolve.push(block)
                self._commit_block(block.block_id)
                self._committing = True
            else:
                LOGGER.info('Failed consensus check: %s', block.block_id.hex())
                self._fail_block(block.block_id)

    def _handle_invalid_block(self, block_id):
        self._validating_blocks.discard(block_id)
        block = self._get_block(block_id)
        LOGGER.info("HANDLING INVALID BLOCK %s", block)
        self._fail_block(block.block_id)

    def _handle_committed_block(self, block_id):
        LOGGER.info(
            'Chain head updated to %s, abandoning block in progress',
            block_id.hex())

        self._cancel_block()
        self._building = False
        self._published = False
        self._committing = False
        self._process_pending_forks()
        self._waiting = None

    def _handle_peer_msgs(self, msg):
        LOGGER.info(msg="HANDLING PEER MSG")
        consensus_msg = ValidatorInfo()
        consensus_msg.ParseFromString(msg[0].content)
        # signer_id = msg[0].header.signer_id.hex()
        validators = self._validators_info_dict.get(consensus_msg.previous_id, None)
        if validators is None:
            self._validators_info_dict[consensus_msg.previous_id] = [consensus_msg]
        elif consensus_msg not in validators:
            self._validators_info_dict[consensus_msg.previous_id].append(consensus_msg)

    def _handle_peer_connected(self, msg):
        LOGGER.info(msg="HANDLING PEER CONNECTED")
        pass

    def _handle_peer_disconnected(self, msg):
        LOGGER.info(msg="HANDLING PEER DISCONNECTED")
        pass

    # Wrapper around service.broadcast to be used for consensus related msgs
    def _broadcast(self, msg):
        self._service.broadcast(message_type=str(Message.CONSENSUS_NOTIFY_PEER_MESSAGE),
                                payload=msg.SerializeToString())

    # Wrapper around service.send_to to be used for consensus related msgs
    def _send_to(self, peer, msg):
        self._service.send_to(receiver_id=peer.encode('utf-8'), message_type=str(Message.CONSENSUS_NOTIFY_PEER_MESSAGE),
                              payload=msg.SerializeToString())
