"""
对区块的一些封装
"""

import os
import logging
import sawtooth_signing as signing
from sawtooth_signing import CryptoFactory
from sawtooth_signing.secp256k1 import Secp256k1PrivateKey

from sawtooth_sdk.consensus.exceptions import UnknownBlock
from sawtooth_sdk.messaging.stream import Stream
from sawtooth_sdk.protobuf.batch_pb2 import Batch
from sawtooth_sdk.protobuf.batch_pb2 import BatchHeader
from sawtooth_sdk.protobuf.client_batch_submit_pb2 import ClientBatchSubmitRequest
from sawtooth_sdk.protobuf.client_batch_submit_pb2 import ClientBatchSubmitResponse
from sawtooth_sdk.protobuf.client_block_pb2 import ClientBlockGetByTransactionIdRequest
from sawtooth_sdk.protobuf.client_block_pb2 import ClientBlockGetResponse
from sawtooth_sdk.protobuf.block_pb2 import BlockHeader
from sawtooth_sdk.protobuf.consensus_pb2 import ConsensusBlock
from sawtooth_sdk.protobuf.validator_pb2 import Message

LOGGER = logging.getLogger(__name__)


# 对共识区块的封装
class PoTEBlock:
    def __init__(self, block):
        self.block_id = block.block_id
        self.previous_id = block.previous_id
        self.signer_id = block.signer_id
        self.block_num = block.block_num
        self.payload = block.payload
        self.summary = block.summary

        identifier = block.block_id.hex()
        previous_block_id = block.previous_id.hex()
        signer_public_key = block.signer_id.hex()

        self.identifier = identifier
        self.previous_block_id = previous_block_id
        self.signer_public_key = signer_public_key

        self.header = _DummyHeader(
            consensus=block.payload,
            signer_public_key=signer_public_key,
            previous_block_id=previous_block_id
        )

        self.state_root_hash = block.block_id

    def __str__(self):
        return (
                "Block("
                + ", ".join([
            "block_num: {}".format(self.block_num),
            "block_id: {}".format(self.block_id.hex()),
            "previous_id: {}".format(self.previous_id.hex()),
            "signer_id: {}".format(self.signer_id.hex()),
            "payload: {}".format(self.payload),
            "summary: {}".format(self.summary.hex()),
        ])
                + ")"
        )


# 简化区块头
class _DummyHeader:
    def __init__(self, consensus, signer_public_key, previous_block_id):
        self.consensus = consensus
        self.signer_public_key = signer_public_key
        self.previous_block_id = previous_block_id


# 加载私钥，获得签名对象
def _load_identity_signer(key_dir, key_name):
    """Loads a private key from the key directory, based on a validator's identity."""
    key_path = os.path.join(key_dir, '{}.priv'.format(key_name))

    if not os.path.exists(key_path):
        raise Exception("No such signing key file: {}".format(key_path))
    if not os.access(key_path, os.R_OK):
        raise Exception("Key file is not readable: {}".format(key_path))

    LOGGER.info('Loading signing key: %s', key_path)
    try:
        with open(key_path, 'r') as key_file:
            private_key_str = key_file.read().strip()
    except IOError as e:
        raise Exception("Could not load key file: {}".format(str(e))) from e

    try:
        private_key = Secp256k1PrivateKey.from_hex(private_key_str)
    except signing.ParseError as e:
        raise Exception("Invalid key in file {}: {}".format(key_path, str(e))) from e

    context = signing.create_context('secp256k1')
    crypto_factory = CryptoFactory(context)
    return crypto_factory.new_signer(private_key)


# 加载私钥，获得私钥字节流
def _load_identify_key(key_dir, key_name):
    key_path = os.path.join(key_dir, '{}.priv'.format(key_name))

    if not os.path.exists(key_path):
        raise Exception("No such signing key file: {}".format(key_path))
    if not os.access(key_path, os.R_OK):
        raise Exception("Key file is not readable: {}".format(key_path))

    LOGGER.info('Loading signing key: %s', key_path)
    try:
        with open(key_path, 'r') as key_file:
            private_key_str = key_file.read().strip()
    except IOError as e:
        raise Exception("Could not load key file: {}".format(str(e))) from e

    try:
        private_key = Secp256k1PrivateKey.from_hex(private_key_str)
    except signing.ParseError as e:
        raise Exception("Invalid key in file {}: {}".format(key_path, str(e))) from e

    return private_key.as_bytes()
