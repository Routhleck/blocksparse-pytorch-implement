import torch


class BlockSparseMatrix:
    def __init__(self, dense_data, shape, block_shape=(32, 32)):
        self.int_type = torch.int32
        # dense矩阵dense_data
        self.dense_data = dense_data
        # 将有数据的块变为以block_size为宽的data串
        self.data = None
        # dense_data的形状, 例如[64, 64]
        self.shape = shape
        # block形状, 例如[32, 32]
        self.block_shape = torch.Size(block_shape)
        # cols_a
        self.cols_a = None
        # row_start_ends_a
        self.row_start_ends_a = None
        # rows_b
        self.rows_b = None
        # col_start_ends_b
        self.col_start_ends_b = None
        # block_mask布尔矩阵, 此block中只要有一个值有数据的为True, 否则为False, 形状为block_count[0] * block_count[1], 类似数组
        self.block_mask = None
        # block_count总共block的数量, 例如[2, 2]
        self.block_count = None
        # n_blocks
        self.n_blocks = None
        self.init_param()

    # 根据data初始化cols_a, row_start_ends_a, rows_b, col_start_ends_b, block_mask
    def init_param(self):
        self.block_count = self.get_block_count()
        self.block_mask = self.get_mask()
        self.n_blocks = self.get_n_blocks()
        self.data = self.get_data()
        blocks = self.get_indices()
        pass

    # 初始化data
    def get_data(self, device="cuda"):
        """
        将有数据的块变为以block_size为宽的data串
        """
        data = torch.zeros(
            (self.n_blocks * self.block_shape[0], self.block_shape[1]),
            dtype=torch.float,
            device=device,
        )
        """
        根据block_mask将dense_data中的数据填充到data中, n_blocks为有数据的block的数量, block_shape为block的形状,
        所以data为(n_blocks * block_shape[0], block_shape[1])的矩阵, 将dense_data中的数据根据block_mask填充到data中
        """
        assignment_count = 0
        for i in range(self.block_count[0]):
            for j in range(self.block_count[1]):
                if self.block_mask[i, j]:
                    data[assignment_count * self.block_shape[0]:(assignment_count + 1) * self.block_shape[0],
                    :] = self.dense_data[i * self.block_shape[0]:(i + 1) * self.block_shape[0],
                         j * self.block_shape[1]:(j + 1) * self.block_shape[1]]
                    assignment_count += 1
        return data

    # 获取总共block的数量
    def get_block_count(self):
        """
        公式为：block_count = ((shape[0] // block_shape[0]), (shape[1] // block_shape[1]))
        """
        return torch.Size((self.shape[0] // self.block_shape[0], self.shape[1] // self.block_shape[1]))

    # 获取需要的block的数量
    def get_n_blocks(self):
        """
        公式为：n_blocks = block_mask.sum()
        """
        return self.block_mask.sum()

    # 获取mask
    def get_mask(self):
        """
        mask为布尔矩阵, 此block中只要有一个值有数据的为True, 否则为False, 形状为(block_count[0], block_count[1]), 类似数组
        需要遍历每个block, 如果有数据则将对应的mask置为True
        """
        X, Y = self.block_count
        block_mask = torch.zeros(X * Y, dtype=torch.bool, device=self.dense_data.device)

        # 遍历每个block
        for i in range(X):
            for j in range(Y):
                # 如果有数据, 则将对应的mask置为True
                if self.dense_data[i * self.block_shape[0]:(i + 1) * self.block_shape[0],
                   j * self.block_shape[1]:(j + 1) * self.block_shape[1]].sum() != 0:
                    block_mask[i * Y + j] = True
        # 将block_mask变为(X, Y)的形状
        block_mask = block_mask.reshape(X, Y)
        return block_mask

    def get_indices_(self, block_ptr, nnzt, transpose_indices):
        device = self.block_mask.device
        X, Y = self.block_count

        rows = nnzt[0]
        cols = nnzt[1]

        if transpose_indices:
            block_indices = torch.zeros(X * Y, dtype=torch.long, device=device)
            """
            positions中的每个元素都是一个表示非零元素在block_mask中的位置的整数
            它的值等于它所在的块的索引加上非零元素在块内的索引
            可以将其理解为将一个二维坐标(row, col)转换成一个一维的位置索引
            就是block_ptr??
            """
            positions = rows * Y + cols
            """
            将block_indices在positions位置上的值设置为block_ptr + 1
            Set the index of used blocks at the used blocks positions : rest will stay zero
            Add 1 temporarily to use 0 as a special value
            """
            block_indices[positions] = block_ptr + 1
            """
            将block_indices变为(X, Y)的形状, 然后转置, 再变为(X * Y)的形状
            Reorganize the indexes with transposed ordering
            """
            block_indices = block_indices.reshape(X, Y).t().reshape(X * Y)
            """
            将block_indices中的值减1, 将block_ptr变为(X * Y)的形状
            Only keeps the non zero, and substract 1 to find back the right block index
            """
            block_ptr = block_indices[torch.nonzero(block_indices, as_tuple=False)] - 1
            # Remove spurious dimension, 将最后一维中的尺寸为1的维度去掉
            block_ptr = block_ptr.squeeze(-1)

            X, Y = Y, X

            rows = cols

            nnztt = torch.nonzero(self.block_mask.t(), as_tuple=False)
            cols = nnztt[:, 1]
        """
        计算CSR格式中的行偏移(row offsets)数组
        row_start_ends 是一个长度为 X+1 的零向量
        其中第 i 个元素表示分块矩阵的前 i 行(列)在 CSR格式中所占据的位置。
        """
        row_start_ends = torch.zeros(X + 1, dtype=torch.long, device=device)
        """
        将row_start_ends在rows位置上的值加1
        """
        row_start_ends.index_add_(
            0,
            rows + 1,
            torch.ones(rows.shape[0], dtype=torch.long, device=device),
        )
        """
        将row_start_ends中的值累加, 得到CSR格式中的行偏移(row offsets)数组
        cumsum(0)是沿着第0维（行）计算累积和，将每个元素与前面的所有元素相加得到新的元素
        """
        row_start_ends = row_start_ends.cumsum(0).to(dtype=self.int_type)
        """
        将cols和block_ptr拼接在一起, 形成一个 (n, 2) 的矩阵
        """
        cols = torch.stack([cols, block_ptr], dim=1).to(dtype=self.int_type)

        return cols, row_start_ends

    # 获取index
    def get_indices(self, block_ptr=None, device="cuda"):
        """
        torch.nonzero(input, *, as_tuple=False) 是一个 PyTorch 的函数，
        用于返回一个张量中所有非零元素的位置索引。
        在本代码中，torch.nonzero 被用来寻找稀疏矩阵中所有非零元素的位置
        block_mask is a boolean mask that gives the block places
        block_ptr, if not None, gives the block position in data for each element of cols_a, otherwise
        assume that the content of block_ptr is just from 0..n_blocks
        Used to recycle blocks
        """

        # 获取block_mask中所有非零元素的位置索引, nnz refer to Nonzero elements
        nnz = torch.nonzero(self.block_mask, as_tuple=False)
        # 若没有block_ptr, 则将block_ptr初始化为从0到n_blocks的数组
        if block_ptr is None:
            block_ptr = torch.arange(0, nnz.shape[0], device=self.dense_data.device)

        # Sort the nnz according to block_ptr to build the self.blocks
        # data used by matmul_with_output_sparse_support_
        _, indices = block_ptr.sort()
        blocks = nnz[indices]
        """
        将blocks这个二维张量先进行水平翻转（即把行变成列，把列变成行）
        再将其展平成一维张量。这样做的目的是将所有非零元素的行坐标和列坐标交替排列在一维张量中
        应该是要使用register_buffer()来注册这个张量，但是这里没有使用
        """
        blocks = blocks.flip(-1).flatten().to(dtype=self.int_type)

        nnzt = nnz.transpose(0, 1)
        self.cols_a, self.row_start_ends_a = self.get_indices_(block_ptr, nnzt, False)
        self.rows_b, self.col_start_ends_b = self.get_indices_(block_ptr, nnzt, True)

        return blocks

    def flatten_first_dims(self, dense_a):
        pass

    def unflatten_first_dims(self, result, info):
        pass

    def matmul_(self, dense_a):
        pass

    def matmul(self, dense_a):
        pass
