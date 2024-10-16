"""
节点集群
"""


# 节点类
class Node:
    def __init__(self, node_type, node_ip, node_port):
        self.type = node_type
        self.ip = node_ip
        self.port = node_port

