"""
命令行接口。这里主要是为了方便测试。如果要直接代码调用，得去 es_client.py 那。
"""

from __future__ import print_function
import argparse
import getpass
import logging
import os
import traceback
import sys
import pkg_resources
from colorlog import ColoredFormatter
from es_exception import ESException
from es_client import ESClient

# 应用名和默认URL
DISTRIBUTION_NAME = 'edge-storage'
DEFAULT_URL = 'http://127.0.0.1:8008'
LOGGER = logging.getLogger(__name__)


# 创建控制台日志处理器
def create_console_handler(verbose_level):
    clog = logging.StreamHandler()
    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s %(levelname)-8s%(module)s]%(reset)s "
        "%(white)s%(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        })

    clog.setFormatter(formatter)
    if verbose_level == 0:
        clog.setLevel(logging.WARN)
    elif verbose_level == 1:
        clog.setLevel(logging.INFO)
    else:
        clog.setLevel(logging.DEBUG)

    return clog


# 设置日志记录器
def setup_loggers(verbose_level):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(create_console_handler(verbose_level))


# 创建游戏的子命令解析器
def add_create_parser(subparsers, parent_parser):
    parser = subparsers.add_parser(
        'create',
        help='Creates a new edge storage contract',
        parents=[parent_parser])

    parser.add_argument(
        'participator',
        type=str,
        help='participants in the contract. if you have key-dir-pub, you can fill in any value at this point')
    parser.add_argument(
        'rule',
        type=str,
        help='rule in the contract')
    parser.add_argument(
        'profit',
        type=str,
        help='profit in the contract')
    parser.add_argument(
        'data_hash',
        type=str,
        help='data_hash in the contract')
    parser.add_argument(
        '--username',
        type=str,
        help="identify name of user's private key file")
    parser.add_argument(
        '--url',
        type=str,
        help='specify URL of REST API')
    parser.add_argument(
        '--key-dir',
        type=str,
        help="identify directory of user's private key file")
    parser.add_argument(
        '--participator-name',
        type=str,
        help="identify directory of participator's username")
    parser.add_argument(
        '--wait',
        nargs='?',
        const=sys.maxsize,
        type=int,
        help='set time, in seconds, to wait for game to commit')


#  签名子命令解析器
def add_sign_parser(subparsers, parent_parser):
    parser = subparsers.add_parser(
        'sign',
        help='Sign the specified contract',
        parents=[parent_parser])
    parser.add_argument(
        '--url',
        type=str,
        help='specify URL of REST API')
    parser.add_argument(
        'contract_hash',
        type=str,
        help='unique identifier for the contract')
    parser.add_argument(
        'location',
        type=str,
        help='If you are the creator signing the parameter is 0, if you are the participant signing the parameter is 1')
    parser.add_argument(
        '--username',
        type=str,
        help="identify name of user's private key file")
    parser.add_argument(
        '--key-dir',
        type=str,
        help="identify directory of creator's private key file")
    parser.add_argument(
        '--wait',
        nargs='?',
        const=sys.maxsize,
        type=int,
        help='set time, in seconds, to wait for game to commit')


def add_confirm_parser(subparsers, parent_parser):
    parser = subparsers.add_parser(
        'confirm',
        help='Confirm the trading result of the specified contract',
        parents=[parent_parser])
    parser.add_argument(
        '--url',
        type=str,
        help='specify URL of REST API')
    parser.add_argument(
        'contract_hash',
        type=str,
        help='unique identifier for the contract')
    parser.add_argument(
        'location',
        type=str,
        help='If you are the creator confirming the parameter is 0, if you are the '
             'participant confirming the parameter is 1')
    parser.add_argument(
        '--username',
        type=str,
        help="identify name of user's private key file")
    parser.add_argument(
        '--key-dir',
        type=str,
        help="identify directory of user's private key file")
    parser.add_argument(
        '--wait',
        nargs='?',
        const=sys.maxsize,
        type=int,
        help='set time, in seconds, to wait for game to commit')


def add_delete_parser(subparsers, parent_parser):
    parser = subparsers.add_parser(
        'delete',
        help='Sign the specified contract',
        parents=[parent_parser])
    parser.add_argument(
        '--url',
        type=str,
        help='specify URL of REST API')
    parser.add_argument(
        'contract_hash',
        type=str,
        help='unique identifier for the contract')
    parser.add_argument(
        '--username',
        type=str,
        help="identify name of user's private key file")
    parser.add_argument(
        '--key-dir',
        type=str,
        help="identify directory of creator's private key file")
    parser.add_argument(
        '--wait',
        nargs='?',
        const=sys.maxsize,
        type=int,
        help='set time, in seconds, to wait for game to commit')


def add_allow_transfer_parser(subparsers, parent_parser):
    parser = subparsers.add_parser(
        'allow_transfer',
        help='Sign the specified contract',
        parents=[parent_parser])
    parser.add_argument(
        '--url',
        type=str,
        help='specify URL of REST API')
    parser.add_argument(
        'contract_hash',
        type=str,
        help='unique identifier for the contract')
    parser.add_argument(
        '--username',
        type=str,
        help="identify name of user's private key file")
    parser.add_argument(
        '--key-dir',
        type=str,
        help="identify directory of creator's private key file")
    parser.add_argument(
        '--wait',
        nargs='?',
        const=sys.maxsize,
        type=int,
        help='set time, in seconds, to wait for game to commit')


def add_not_allow_transfer_parser(subparsers, parent_parser):
    parser = subparsers.add_parser(
        'not_allow_transfer',
        help='Sign the specified contract',
        parents=[parent_parser])
    parser.add_argument(
        '--url',
        type=str,
        help='specify URL of REST API')
    parser.add_argument(
        'contract_hash',
        type=str,
        help='unique identifier for the contract')
    parser.add_argument(
        '--username',
        type=str,
        help="identify name of user's private key file")
    parser.add_argument(
        '--key-dir',
        type=str,
        help="identify directory of creator's private key file")
    parser.add_argument(
        '--wait',
        nargs='?',
        const=sys.maxsize,
        type=int,
        help='set time, in seconds, to wait for game to commit')


# 显示合约详情的子命令解析器
def add_show_parser(subparsers, parent_parser):
    parser = subparsers.add_parser(
        'show',
        help='show an edge storage contract',
        parents=[parent_parser])
    parser.add_argument(
        '--url',
        type=str,
        help='specify URL of REST API')
    parser.add_argument(
        'contract_hash',
        type=str,
        help='unique identifier for the contract')


# 创建父解析器，用于全局参数，如verbose和version
def create_parent_parser(prog_name):
    parent_parser = argparse.ArgumentParser(prog=prog_name, add_help=False)
    parent_parser.add_argument(
        '-v', '--verbose',
        action='count',
        help='enable more verbose output')

    try:
        version = pkg_resources.get_distribution(DISTRIBUTION_NAME).version
    except pkg_resources.DistributionNotFound:
        version = 'UNKNOWN'

    parent_parser.add_argument(
        '-V', '--version',
        action='version',
        version=(DISTRIBUTION_NAME + ' (Hyperledger Sawtooth) version {}')
        .format(version),
        help='display version information')

    return parent_parser


# 创建解析器和处理命令
def create_parser(prog_name):
    parent_parser = create_parent_parser(prog_name)

    parser = argparse.ArgumentParser(
        description='Provides subcommands to create or operate contract by sending transactions.',
        parents=[parent_parser])

    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    subparsers.required = True

    add_create_parser(subparsers, parent_parser)
    add_show_parser(subparsers, parent_parser)
    add_sign_parser(subparsers, parent_parser)
    add_confirm_parser(subparsers, parent_parser)
    add_delete_parser(subparsers, parent_parser)
    add_allow_transfer_parser(subparsers, parent_parser)
    add_not_allow_transfer_parser(subparsers, parent_parser)

    return parser


# 处理show命令的函数
def do_show(args):
    contract_hash = args.contract_hash

    url = _get_url(args)

    client = ESClient(base_url=url, keyfile=None)

    data = client.show(contract_hash)

    if data is not None:
        creator, participator, rule, profit, data_hash, is_transfer, is_signed_by_creator, \
            is_signed_by_participator, execution_result, confirmation_result, allow_transfer = (data.decode()
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

    else:
        raise ESException("Contract not found: {}".format(contract_hash))


# 处理create命令的函数
def do_create(args):
    participator = args.participator

    if args.participator_name is not None:
        participator = _get_pub(args.participator_name)

    rule = args.rule
    profit = args.profit
    data_hash = args.data_hash

    url = _get_url(args)
    keyfile = _get_keyfile(args)

    client = ESClient(base_url=url, keyfile=keyfile)

    if args.wait and args.wait > 0:
        response = client.create(participator, rule, profit, data_hash, wait=args.wait)
    else:
        response = client.create(participator, rule, profit, data_hash)

    print("Response: {}".format(response))


# 处理sign命令的函数
def do_sign(args):
    contract_hash = args.contract_hash
    signature_location = args.location

    print(contract_hash)
    url = _get_url(args)
    keyfile = _get_keyfile(args)
    client = ESClient(base_url=url, keyfile=keyfile)
    if args.wait and args.wait > 0:
        response = client.sign(contract_hash, signature_location, wait=args.wait)
    else:
        response = client.sign(contract_hash, signature_location)

    print("Response: {}".format(response))


def do_confirm(args):
    contract_hash = args.contract_hash
    signature_location = args.location

    url = _get_url(args)
    keyfile = _get_keyfile(args)
    client = ESClient(base_url=url, keyfile=keyfile)

    if args.wait and args.wait > 0:
        response = client.confirm(contract_hash, signature_location, wait=args.wait)
    else:
        response = client.confirm(contract_hash, signature_location)

    print("Response: {}".format(response))


def do_delete(args):
    contract_hash = args.contract_hash

    url = _get_url(args)
    keyfile = _get_keyfile(args)
    client = ESClient(base_url=url, keyfile=keyfile)

    if args.wait and args.wait > 0:
        response = client.delete(contract_hash, wait=args.wait)
    else:
        response = client.delete(contract_hash)

    print("Response: {}".format(response))


def do_allow_transfer(args):
    contract_hash = args.contract_hash

    url = _get_url(args)
    keyfile = _get_keyfile(args)
    client = ESClient(base_url=url, keyfile=keyfile)

    if args.wait and args.wait > 0:
        response = client.allow_transfer(contract_hash, wait=args.wait)
    else:
        response = client.allow_transfer(contract_hash)

    print("Response: {}".format(response))


def do_not_allow_transfer(args):
    contract_hash = args.contract_hash

    url = _get_url(args)
    keyfile = _get_keyfile(args)
    client = ESClient(base_url=url, keyfile=keyfile)

    if args.wait and args.wait > 0:
        response = client.not_allow_transfer(contract_hash, wait=args.wait)
    else:
        response = client.not_allow_transfer(contract_hash)

    print("Response: {}".format(response))


# 获取URL的辅助函数
def _get_url(args):
    return DEFAULT_URL if args.url is None else args.url


# 获取密钥文件的辅助函数
def \
        _get_keyfile(args):
    username = getpass.getuser() if args.username is None else args.username
    home = os.path.expanduser("~")
    key_dir = os.path.join(home, ".sawtooth", "keys")
    return '{}/{}.priv'.format(key_dir, username)


# 获得公钥的辅助函数
def _get_pub(user):
    home = os.path.expanduser("~")
    key_dir = os.path.join(home, ".sawtooth", "keys")
    keyfile = '{}/{}.pub'.format(key_dir, user)

    try:
        with open(keyfile) as fd:
            pub_key_str = fd.read().strip()
    except OSError as err:
        # 读取文件失败时抛出异常
        raise ESException(
            'Failed to read private key {}: {}'.format(keyfile, str(err))) from err

    return pub_key_str


# 主函数
def main(prog_name=os.path.basename(sys.argv[0]), args=None):
    if args is None:
        args = sys.argv[1:]
    parser = create_parser(prog_name)
    args = parser.parse_args(args)
    if args.verbose is None:
        verbose_level = 0
    else:
        verbose_level = args.verbose
    setup_loggers(verbose_level=verbose_level)

    if args.command == 'create':
        do_create(args)
    elif args.command == 'show':
        do_show(args)
    elif args.command == 'sign':
        do_sign(args)
    elif args.command == 'confirm':
        do_confirm(args)
    elif args.command == 'delete':
        do_delete(args)
    elif args.command == 'allow_transfer':
        do_allow_transfer(args)
    elif args.command == 'not_allow_transfer':
        do_not_allow_transfer(args)
    else:
        raise ESException("invalid command: {}".format(args.command))


# 主函数的包装器
def main_wrapper():
    try:
        main()
    except ESException as err:
        print("Error: {}".format(err), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    except SystemExit as err:
        raise err
    except BaseException as err:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main_wrapper()
