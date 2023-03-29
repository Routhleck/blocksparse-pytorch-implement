import torch


class BlockSparseMatrix:
    def __init__(self, dense_data, shape, block_shape=(32, 32)):
        self.int_type = torch.int32
        # dense矩阵dense_data
        self.dense_data = dense_data
        # 将有数据的块变为以block_size为宽的data串
        self.data = None
        # data的形状, 例如[64, 64]
        self.shape = shape
        # block形状, 例如[32, 32]
        self.block_shape = block_shape
        # cols_a
        self.cols_a = None
        # row_start_ends_a
        self.row_start_ends_a = None
        # rows_b
        self.rows_b = None
        # col_start_ends_b
        self.col_start_ends_b = None
        # block_mask
        self.block_mask = None
        # block_count
        self.block_count = None

    # 根据data初始化cols_a, row_start_ends_a, rows_b, col_start_ends_b, block_mask
    def init_param(self):
        self.block_count = self.get_block_count()
        self.mask = self.get_mask()
        self.cols_a, self.row_start_ends_a, self.rows_b, self.col_start_ends_b = self.get_index()
        pass

    # 获取block的数量
    def get_block_count(self):
        """
        公式为：block_count = ((shape[0] // block_shape[0]), (shape[1] // block_shape[1]))
        """
        return [(self.shape[0] // self.block_shape[0]), (self.shape[1] // self.block_shape[1])]
    # 获取mask
    def get_mask(self):
        """
        mask为布尔矩阵, 此block中只要有一个值有数据的为True, 否则为False, 形状为block_count
        """
        pass

    def get_index(self):
        pass

    def flatten_first_dims(self, dense_a):
        pass

    def unflatten_first_dims(self, result, info):
        pass

    def matmul_(self, dense_a):
        pass

    def matmul(self, dense_a):
        pass
