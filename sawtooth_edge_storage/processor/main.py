"""
交易处理器的主入口，负责启动和配置交易处理器
"""

import sys
import os
import argparse
from importlib_metadata import version, PackageNotFoundError
# from sawtooth_edge_storage.processor.handler import ESTransactionHandler
from handler import ESTransactionHandler

from sawtooth_sdk.processor.core import TransactionProcessor
from sawtooth_sdk.processor.log import init_console_logging
from sawtooth_sdk.processor.log import log_configuration
from sawtooth_sdk.processor.config import get_log_dir

# 定义软件包名称
DISTRIBUTION_NAME = 'sawtooth-edge-storage'


# 解析命令行参数的函数
def parse_args(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # 定义连接参数
    parser.add_argument('-C', '--connect', help='Endpoint for the validator connection',
                        default='tcp://localhost:4004')

    # 定义日志级别参数
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase output sent to stderr')

    # 获取版本信息
    try:
        package_version = version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        package_version = 'UNKNOWN'

    # 添加版本信息参数
    parser.add_argument('-V', '--version', action='version',
                        version=(DISTRIBUTION_NAME + ' (Hyperledger Sawtooth) version {}').format(package_version),
                        help='print version information')

    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    opts = parse_args(args)
    processor = None

    try:

        processor = TransactionProcessor(url=opts.connect)

        # 配置日志
        log_dir = get_log_dir()
        log_configuration(log_dir=log_dir, name="es-" + str(processor.zmq_id)[2:-1])

        # 初始化控制台日志
        init_console_logging(verbose_level=opts.verbose)

        # 添加交易处理器handler
        handler = ESTransactionHandler()
        processor.add_handler(handler)

        # 启动交易处理器
        processor.start()

    # 处理中断和异常
    except KeyboardInterrupt:
        pass
    except Exception as e:  # pylint: disable=broad-except
        print("Error: {}".format(e))
    finally:
        if processor is not None:
            processor.stop()


if __name__ == "__main__":
    main()
