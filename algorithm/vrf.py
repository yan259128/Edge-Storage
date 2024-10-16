"""
VRF函数的实现
"""
import hashlib
import sys

if sys.version_info[0] != 3 or sys.version_info[1] < 7:
    print("Requires Python v3.7+")
    sys.exit()


# Public API

# ECVRF Proving
def ecvrf_prove(sk, m):
    """
    输入：
        sk-VRF私钥（32字节）
        m-输入m，一个八位字节字符串
    输出：
        （“VALID”，vrf_proof）-其中vrf_proof是VRF证明，长度为ptLen+n+qLen的八位字节字符串
        （80）字节，或（“无效”，[]）故障时
    """
    # 1
    secret_scalar_x = _get_secret_scalar(sk)
    public_key_y = get_public_key(sk)

    # 2
    h = _ecvrf_hash_to_curve_elligator2_25519(SUITE_STRING, public_key_y, m)
    if h == "INVALID":
        return "INVALID", []

    # 3
    h_string = _decode_point(h)
    if h_string == "INVALID":
        return "INVALID", []

    # 4
    gamma = _scalar_multiply(p=h_string, e=secret_scalar_x)

    # 5
    k = _ecvrf_nonce_generation_rfc8032(sk, h)

    # 6
    k_b = _scalar_multiply(p=BASE, e=k)
    k_h = _scalar_multiply(p=h_string, e=k)
    c = _ecvrf_hash_points(h_string, gamma, k_b, k_h)

    # 7
    s = (k + c * secret_scalar_x) % ORDER

    # 8
    vrf_proof = _encode_point(gamma) + int.to_bytes(c, 16, 'little') + int.to_bytes(s, 32, 'little')

    if 'test_dict' in globals():
        _assert_and_sample(['secret_scalar_x', 'public_key_y', 'h', 'gamma', 'k_b', 'k_h', 'vrf_proof'],
                           [secret_scalar_x.to_bytes(32, 'little'), public_key_y, h, _encode_point(gamma),
                            _encode_point(k_b), _encode_point(k_h), vrf_proof])

    # 9. Output vrf_proof
    return "VALID", vrf_proof


# ECVRF Proof To Hash
def ecvrf_proof_to_hash(vrf_proof):
    """
    输入：
        vrf_proof-VRF证明，长度为ptLen+n+qLen（80）字节的八位字节字符串
    输出：
        （“VALID”，beta_string）其中beta_string是VRF哈希输出，八位字节字符串
        长度为hLen（64）字节，或（“无效”，[]）（故障时）
    """
    # 1
    d = _ecvrf_decode_proof(vrf_proof)

    # 2
    if d == "INVALID":
        return "INVALID", []

    # 3
    gamma, c, s = d

    # 4
    three_string = bytes([0x03])

    # 5
    cofactor_gamma = _scalar_multiply(p=gamma, e=COFACTOR)  # Curve cofactor
    beta_string = _hash(SUITE_STRING + three_string + _encode_point(cofactor_gamma))

    if 'test_dict' in globals():
        _assert_and_sample(['beta_string'], [beta_string])

    # 6
    return "VALID", beta_string


# ECVRF Verifying
def ecvrf_verify(y, vrf_proof, m):
    """
    输入：
        y-公钥，以字节表示的EC点
        vrf_proof-VRF证明，长度为ptLen+n+qLen（80）字节的八位字节字符串
        m-VRF输入，八位字节字符串
    输出：
        （“VALID”，beta_string），其中beta_string是VRF哈希输出，八位字节字符串
        长度为hLen（64）字节；或（“无效”，[]）发生故障时
    """
    # 请注意，API调用程序需要验证返回的beta_string是否为预期值，
    # 这极有可能导致错误/疏忽（例如检查“VALID”而不是实际值）。
    # 通过传递预期的beta_string并获得更简单的通过/失败响应，
    # 可以更好地服务于生产代码。

    # 1
    d = _ecvrf_decode_proof(vrf_proof)

    # 2
    if d == "INVALID":
        return "INVALID", []

    # 3
    gamma, c, s = d

    # 4
    h = _ecvrf_hash_to_curve_elligator2_25519(SUITE_STRING, y, m)
    if h == "INVALID":
        return "INVALID", []

    # 5
    y_point = _decode_point(y)
    h_point = _decode_point(h)
    if y_point == "INVALID" or h_point == "INVALID":
        return "INVALID", []
    s_b = _scalar_multiply(p=BASE, e=s)
    c_y = _scalar_multiply(p=y_point, e=c)
    nc_y = [PRIME - c_y[0], c_y[1]]
    u = _edwards_add(s_b, nc_y)

    # 6
    s_h = _scalar_multiply(p=h_point, e=s)
    c_g = _scalar_multiply(p=gamma, e=c)
    nc_g = [PRIME - c_g[0], c_g[1]]
    v = _edwards_add(nc_g, s_h)

    # 7
    cp = _ecvrf_hash_points(h_point, gamma, u, v)

    if 'test_dict' in globals():
        _assert_and_sample(['h', 'u', 'v'], [h, _encode_point(u), _encode_point(v)])

    # 8. 如果c等于cp, 返回 ("VALID", EC_vrf_proof_to_hash(vrf_proof)); 否则返回 "INVALID"
    if c == cp:
        return ecvrf_proof_to_hash(vrf_proof)
    else:
        return "INVALID", []


def get_public_key(sk):
    """
    计算公钥并将其作为编码点字符串返回（字节）
    """
    secret_int = _get_secret_scalar(sk)
    public_point = _scalar_multiply(p=BASE, e=secret_int)
    public_string = _encode_point(public_point)
    return public_string


# 内部功能
# ECVRF_hash_to_curve_elligator2_25519
def _ecvrf_hash_to_curve_elligator2_25519(suite_string, y, m):
    """
    输入：
        suite_string-指定ECVRF密码套件的单个八位字节。
        m-要散列的值，一个八位字节字符串
        y-公钥，以字节表示的EC点
    输出：
        H-散列值，G中的有限EC点，或失败时无效
        固定选项：
        p=2^255-19，有限域F的大小，素数，对于爱德华25519和曲线25519曲线
        A=486662，曲线的Montgomery曲线常数25519
        辅因子=8，爱德华25519和曲线25519的辅因子
    """
    assert suite_string == SUITE_STRING
    # 1
    # 2
    one_string = bytes([0x01])

    # 3
    hash_string = _hash(suite_string + one_string + y + m)

    # 4
    r_string = bytearray(hash_string[0:32])

    # 5
    one_twenty_seven_string = 0x7f

    # 6
    r_string[31] = int(r_string[31] & one_twenty_seven_string)

    # 7
    r = int.from_bytes(r_string, 'little')

    # 8
    u = (PRIME - A) * _inverse(1 + 2 * (r ** 2)) % PRIME

    # 9
    w = u * (u ** 2 + A * u + 1) % PRIME

    # 10
    e = pow(w, (PRIME - 1) // 2, PRIME)

    # 11
    final_u = (e * u + (e - 1) * A * TWO_INV) % PRIME
    # 请注意，虽然上面的公式在恒定时间实现中有一定的意义，但该实现并不旨在是恒定时间。因此，它可以大大简化。

    # 12
    y_coordinate = (final_u - 1) * _inverse(final_u + 1) % PRIME

    # 13
    y_string = int.to_bytes(y_coordinate, 32, 'little')

    # 14
    h_prelim = _decode_point(y_string)
    if h_prelim == "INVALID":
        return "INVALID"

    # 15.
    h = _scalar_multiply(p=h_prelim, e=COFACTOR)  # Curve cofactor

    # 16. 返回 H
    h_point = _encode_point(h)

    if 'test_dict' in globals():
        _assert_and_sample(['r', 'w', 'e'],
                           [r_string, int.to_bytes(w, 32, 'little'), int.to_bytes(e, 32, 'little')])

    return h_point


# ECVRF Nonce Generation From RFC 8032
def _ecvrf_nonce_generation_rfc8032(sk, h_string):
    """
    输入：
        sk-以字节表示的ECVRF密钥
        h_string-八位字节字符串
    输出：
        k-介于0和q-1之间的整数
    """
    # 1
    hashed_sk_string = _hash(sk)

    # 2
    truncated_hashed_sk_string = hashed_sk_string[32:]

    # 3
    k_string = _hash(truncated_hashed_sk_string + h_string)

    # 4
    k = int.from_bytes(k_string, 'little') % ORDER

    if 'test_dict' in globals():
        _assert_and_sample(['k'], [k_string])

    return k


# ECVRF Hash Points
def _ecvrf_hash_points(p1, p2, p3, p4):
    """
    输入：
        P1…PM-G中的EC点
    输出：
        c-哈希值，0到2^（8n）-1之间的整数
    """
    # 1
    two_string = bytes([0x02])

    # 2
    string = SUITE_STRING + two_string

    # 3
    string = string + _encode_point(p1) + _encode_point(p2) + _encode_point(p3) + _encode_point(p4)

    # 4
    c_string = _hash(string)

    # 5
    truncated_c_string = c_string[0:16]

    # 6
    c = int.from_bytes(truncated_c_string, 'little')

    # 7. 返回 c
    return c


# ECVRF Decode Proof
def _ecvrf_decode_proof(vrf_proof):
    """
    输入：
        vrf_proof-VRF的证明，八位字节字符串（ptLen+n+qLen八位字节）
    输出：
        “无效”，或伽玛-EC点
        c-0到2^（8n）-1之间的整数
        s-0到2^（8qLen）-1之间的整数
    """
    if len(vrf_proof) != 80:  # ptLen+n+qLen octets = 32+16+32 = 80
        return "INVALID"

    # 1
    gamma_string = vrf_proof[0:32]

    # 2
    c_string = vrf_proof[32:48]

    # 3
    s_string = vrf_proof[48:]

    # 4
    gamma = _decode_point(gamma_string)

    # 5. 如果 Gamma = "INVALID" 返回 "INVALID"
    if gamma == "INVALID":
        return "INVALID"

    # 6
    c = int.from_bytes(c_string, 'little')

    # 7
    s = int.from_bytes(s_string, 'little')

    # 8. 返回 Gamma, c, and s
    return gamma, c, s


def _assert_and_sample(keys, actual):
    """
    输入：
        key-key表示断言值，basename（+'_sample'）表示采样值。
    输出：
        没有一个然后断言actual并分配到全局test_dict
        若键存在，则根据提供的实际值断言dict预期值。
        对实际值进行采样，并将其存储到test_dict中的键+“_Sample”下。
    """
    # 未定义的
    global test_dict
    for key, actual in zip(keys, actual):
        if key in test_dict and actual:
            assert actual == test_dict[key], "{}  actual:{} != expected:{}".format(key, actual.hex(),
                                                                                   test_dict[key].hex())
        test_dict[key + '_sample'] = actual


"""
以下大部分代码都改编自ed25519.py，位于https://ed25519.cr.yp.to/software.html检索日期：2019年12月27日
虽然它效率低下，但它提供了底层数学的极好演示。例如，生产
代码可能会避免通过费马小定理进行反演，因为这是极其昂贵的，大约需要300次场乘法。
"""


def _edwards_add(p, q):
    """
    Edwards曲线点添加
    """
    x1 = p[0]
    y1 = p[1]
    x2 = q[0]
    y2 = q[1]
    x3 = (x1 * y2 + x2 * y1) * _inverse(1 + D * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * _inverse(1 - D * x1 * x2 * y1 * y2)
    return [x3 % PRIME, y3 % PRIME]


def _encode_point(p):
    """
    对包含X的LSB和y的254位的点到字符串进行编码
    """
    return ((p[1] & ((1 << 255) - 1)) + ((p[0] & 1) << 255)).to_bytes(32, 'little')


def _decode_point(s):
    """
    将包含X的LSB和y的254位的字符串解码为点。检查曲线。可能返回 \"INVALID\"
    """
    y = int.from_bytes(s, 'little') & ((1 << 255) - 1)
    x = _x_recover(y)
    if x & 1 != _get_bit(s, BITS - 1):
        x = PRIME - x
    p = [x, y]
    if not _is_on_curve(p):
        return "INVALID"
    return p


def _get_bit(h, i):
    """Return specified bit from string for subsequent testing"""
    h1 = int.from_bytes(h, 'little')
    return (h1 >> i) & 0x01


def _get_secret_scalar(sk):
    """
    计算并返回secret_scalar整数
    """
    h = bytearray(_hash(sk)[0:32])
    h[31] = int((h[31] & 0x7f) | 0x40)
    h[0] = int(h[0] & 0xf8)
    secret_int = int.from_bytes(h, 'little')
    return secret_int


def _hash(message):
    """
    返回任意长度字节消息的64字节SHA512哈希
    """
    return hashlib.sha512(message).digest()


def _inverse(x):
    """
    利用费马小定理求逆
    """
    return pow(x, PRIME - 2, PRIME)


def _is_on_curve(p):
    """
    检查以确认点在曲线上；返回布尔值
    """
    x = p[0]
    y = p[1]
    result = (-x * x + y * y - 1 - D * x * x * y * y) % PRIME
    return result == 0


def _scalar_multiply(p, e):
    """
    标量乘以曲线点
    """
    if e == 0:
        return [0, 1]
    q = _scalar_multiply(p, e // 2)
    q = _edwards_add(q, q)
    if e & 1:
        q = _edwards_add(q, p)
    return q


def _x_recover(y):
    """
    从y坐标恢复x坐标
    """
    xx = (y * y - 1) * _inverse(D * y * y + 1)
    x = pow(xx, (PRIME + 3) // 8, PRIME)
    if (x * x - xx) % PRIME != 0:
        x = (x * II) % PRIME
    if x % 2 != 0:
        x = PRIME - x
    return x


# 常量，其中一些在运行时使用上述例程进行计算/检查
#  https://ed25519.cr.yp.to/python/checkparams.py
SUITE_STRING = bytes([0x04])
BITS = 256
PRIME = 2 ** 255 - 19
ORDER = 2 ** 252 + 27742317777372353535851937790883648493
COFACTOR = 8
TWO_INV = _inverse(2)
II = pow(2, (PRIME - 1) // 4, PRIME)
A = 486662
D = -121665 * _inverse(121666)
BASEy = 4 * _inverse(5)
BASEx = _x_recover(BASEy)
BASE = [BASEx % PRIME, BASEy % PRIME]
assert BITS >= 10
assert 8 * len(_hash("hash input".encode("UTF-8"))) == 2 * BITS
assert pow(2, PRIME - 1, PRIME) == 1
assert PRIME % 4 == 1
assert pow(2, ORDER - 1, ORDER) == 1
assert ORDER >= 2 ** (BITS - 4)
assert ORDER <= 2 ** (BITS - 3)
assert pow(D, (PRIME - 1) // 2, PRIME) == PRIME - 1
assert pow(II, 2, PRIME) == PRIME - 1
assert _is_on_curve(BASE)
assert _scalar_multiply(BASE, ORDER) == [0, 1]
