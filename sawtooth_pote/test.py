"""
测试 vrf
"""

from sawtooth_pote.consensus.validator_registry_pb2 import ValidatorInfo

import sawtooth_signing as signing
from sawtooth_signing import CryptoFactory
from sawtooth_signing.secp256k1 import Secp256k1PrivateKey
# import secrets
import sawtooth_pote.consensus.vrf as vrf

# secret_key = secrets.token_bytes(nbytes=32)
private_key = Secp256k1PrivateKey.new_random().as_bytes()[:32]
public_key = vrf.get_public_key(private_key)

# Alice generates a beta_string commitment to share with Bob
m = b'I bid $100 for the horse named IntegrityChain'
p_status, proof = vrf.ecvrf_prove(private_key, m)
b_status, beta_string = vrf.ecvrf_proof_to_hash(proof)

validator = ValidatorInfo(id='1', proof=proof, previous_id='0',
                          proof_data=m, expectation=0.1,
                          name='validator', public_key=public_key,
                          random=beta_string)
# validator = ValidatorInfo()
print(validator)
