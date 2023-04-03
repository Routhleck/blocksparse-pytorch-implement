import torch


class BlockSparseMatrix(torch.nn.Module):
    def __init__(self, shape, block_shape, dense_data=None, data=None, block_mask=None):
        super().__init__()
        self.int_type = torch.int32
        # 将有数据的块变为以block_size为宽的data串
        self.data = data
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
        self.block_mask = block_mask
        # block_count总共block的数量, 例如[2, 2]
        self.block_count = None
        # n_blocks
        self.n_blocks = None
        if dense_data is not None:
            self.init_param_from_dense(dense_data)
        elif block_mask is not None and data is not None:
            self.init_param_from_mask_and_data(block_mask, data)
        else:
            pass

    # 根据data初始化cols_a, row_start_ends_a, rows_b, col_start_ends_b, block_mask
    def init_param_from_dense(self, dense_data):
        self.block_count = self.get_block_count()
        self.block_mask = self.get_mask(dense_data)
        self.n_blocks = self.get_n_blocks()
        self.data = self.get_data(dense_data)
        blocks = self.get_indices()

        # register
        for name in (
                "blocks",
        ):
            self.register_buffer(name, locals()[name])

    def init_param_from_mask_and_data(self, block_mask, data):
        self.block_count = self.get_block_count()
        self.n_blocks = self.get_n_blocks()
        blocks = self.get_indices()

        for name in (
            "blocks",
        ):
            self.register_buffer(name, locals()[name])

    # 初始化data
    def get_data(self, dense_data, device="cuda"):
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
                    :] = dense_data[i * self.block_shape[0]:(i + 1) * self.block_shape[0],
                         j * self.block_shape[1]:(j + 1) * self.block_shape[1]]
                    assignment_count += 1
        return data

    # 获取总共block的数量
    @staticmethod
    def get_block_count_(shape, block_shape):
        return torch.Size((shape[0] // block_shape[0], shape[1] // block_shape[1]))

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
    def get_mask(self, dense_data):
        """
        mask为布尔矩阵, 此block中只要有一个值有数据的为True, 否则为False, 形状为(block_count[0], block_count[1]), 类似数组
        需要遍历每个block, 如果有数据则将对应的mask置为True
        """
        X, Y = self.block_count
        block_mask = torch.zeros(X * Y, dtype=torch.bool, device=dense_data.device)

        # 遍历每个block
        for i in range(X):
            for j in range(Y):
                # 如果有数据, 则将对应的mask置为True
                if dense_data[i * self.block_shape[0]:(i + 1) * self.block_shape[0],
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
    def get_indices(self, block_ptr=None):
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
            block_ptr = torch.arange(0, nnz.shape[0], device=self.block_mask.device)

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
        if dense_a.dim() < 2:
            raise RuntimeError("Dense matrix should have at least 2 dimensions")
        rewritten_a = dense_a
        if dense_a.dim() > 2:
            # 将原先的前几维合并成一维, 最后一维保持不变
            rewritten_a = dense_a.reshape(-1, dense_a.shape[-1])
        return rewritten_a, dense_a.shape[:-1]

    def unflatten_first_dims(self, result, info):
        shape_start = info
        if len(shape_start) > 1:
            result = result.reshape(*shape_start, result.shape[-1])
        return result

    def matmul_(self, dense_a, transpose):
        import block_sparse_native

        shape_a = list(dense_a.shape)
        shape_b = [self.shape[0], self.shape[1]]
        block_shape = list(self.block_shape)

        if transpose:
            shape_b.reverse()
            block_shape.reverse()

        if shape_a[1] != shape_b[0]:
            raise Exception(
                "Invalid matrices sizes (%d, %d) x (%d, %d)" % (shape_a[0], shape_a[1], shape_b[0], shape_b[1])
            )

        result = torch.zeros((shape_b[1], shape_a[0]), device=dense_a.device)

        if transpose:
            ptr_b = self.row_start_ends_a
            indices_b = self.cols_a
            dim = 0
        else:
            ptr_b = self.col_start_ends_b
            indices_b = self.rows_b
            dim = 1

        assert (shape_a[1] % block_shape[1]) == 0
        assert self.data.is_contiguous()
        assert result.is_contiguous()

        assert ptr_b.is_contiguous()
        assert ptr_b.dtype == self.int_type
        assert indices_b.is_contiguous()
        assert indices_b.dtype == self.int_type

        assert ptr_b.shape[0] == self.block_count[dim] + 1

        if transpose:
            data_b = self.data
        else:
            data = self.data.view(-1, *block_shape)
            data = data.transpose(1, 2)
            data_b = data.reshape(-1, block_shape[1]).contiguous()

        if not dense_a.is_contiguous():
            dense_a = dense_a.contiguous()

        block_sparse_native.blocksparse_matmul_cutlass(
            dense_a,
            True,
            ptr_b,
            indices_b,
            data_b,
            dense_a.shape[0],
            shape_b[1],
            shape_b[0],
            block_shape[1],
            block_shape[0],
            result,
        )
        return result.t()

    def matmul(self, dense_a, transpose=False):
        rewritten_a, info_a = self.flatten_first_dims(dense_a)
        result = self.matmul_(rewritten_a, transpose=transpose)
        result = self.unflatten_first_dims(result, info_a)
        return result

    # 从dense_data构建BlockSparseMatrix类
    @staticmethod
    def from_dense(dense_data, block_shape=(32, 32)):
        shape = dense_data.shape
        return BlockSparseMatrix(shape, block_shape, dense_data=dense_data)

    @staticmethod
    def from_mask_and_data(shape, block_mask, data, block_shape=(32, 32)):
        return BlockSparseMatrix(shape, block_shape, block_mask=block_mask, data=data)

    def to_dense(self):
        # 将稀疏矩阵转换为密集矩阵
        dense_data = torch.zeros(self.shape, dtype=torch.float, device=self.data.device)
        mask_count = 0
        for i in range(self.block_count[0]):
            for j in range(self.block_count[1]):
                if self.block_mask[i, j]:
                    dense_data[
                        i * self.block_shape[0] : (i + 1) * self.block_shape[0],
                        j * self.block_shape[1] : (j + 1) * self.block_shape[1],
                    ] = self.data[mask_count * self.block_shape[0] : (mask_count + 1) * self.block_shape[0], :]
        return dense_data

    @classmethod
    def randn(
        cls,
        shape,
        n_blocks=None,
        block_shape=(32, 32),
        device="cuda",
        positive=False,
    ):
        result = cls.zeros(shape, n_blocks, block_shape, device)
        with torch.no_grad():
            if positive:
                result.data.normal_().abs(result.data)
            else:
                result.data.normal_()
        return result

    @classmethod
    def zeros(
        cls,
        shape,
        n_blocks=None,
        block_shape=(32, 32),
        device="cuda",
    ):
        """
        生成一个全0的稀疏矩阵
        先生成mask的布尔矩阵, mask的shape为shape[0] // block_shape[0] * shape[1] // block_shape[1]
        若n_blocks为None, 则随机取 0 到 shape[0] * shape[1] // (block_shape[0] * block_shape[1])之间个块为True
        若n_blocks不为None, 则随机取n_blocks个块为True
        """
        block_count = cls.get_block_count_(shape=shape,block_shape=block_shape)
        if n_blocks is None:
            n_blocks = torch.randint(0, block_count[0] * block_count[1], (1,), device=device).item()
        else:
            n_blocks = min(n_blocks, block_count[0] * block_count[1])
        block_mask = torch.zeros(block_count, dtype=torch.bool, device=device)
        block_indices = torch.randperm(block_count[0] * block_count[1], device=device)[:n_blocks]
        block_mask.view(-1)[block_indices] = True

        data = torch.zeros((n_blocks * block_shape[0], block_shape[1]), dtype=torch.float, device=device)
        return cls.from_mask_and_data(shape=shape, block_shape=block_shape, block_mask=block_mask, data=data)

    @classmethod
    def ones(
        cls,
        shape,
        n_blocks=None,
        block_shape=(32, 32),
        device="cuda",
        positive=False,
    ):
        result = cls.zeros(shape, n_blocks, block_shape, device)
        with torch.no_grad():
            result.data += 1
        return result

