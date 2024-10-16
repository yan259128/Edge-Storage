"""
存储分叉最新区块的数据结构，由于分叉未实现，暂搁置
"""


class PendingForks:
    def __init__(self):
        self._queue = []  # 存储所有暂未被解决的分叉的最新区块
        self._blocks = {}

    def push(self, block):

        # 如果当前区块的上一个区块不在队列中，则说明产生了一个新的分叉，入队。
        try:
            index = self._queue.index(block.previous_id)
        except ValueError:
            self._queue.insert(0, block.block_id)
            self._blocks[block.block_id] = block
            return

        # 如果存在上一个区块，则说明当前分叉出现了新的区块。将旧的覆盖。
        del self._blocks[block.previous_id]
        self._queue[index] = block.block_id
        self._blocks[block.block_id] = block

    def pop(self):
        try:
            block_id = self._queue.pop()
        except IndexError:
            return None

        return self._blocks.pop(block_id, None)
