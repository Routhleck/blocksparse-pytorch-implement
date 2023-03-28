import torch

class BlockSparseMatrix:
    def __init__(self, data, shape, block_shape = (32, 32)):
        self.int_type = torch.int32
        # dense矩阵data
        self.data = data
        # data的形状
        self.shape = shape
        # block形状
        self.block_shape = block_shape

