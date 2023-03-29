import blocksparse.BlockSparseMatrix as BlockSparseMatrix

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# dense_data为((1, 1, 1, 0), (1, 1, 1, 0), (1, 1, 1, 0), (0, 0, 0, 0))
dense_data = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
print('dense_data:', dense_data)
shape = dense_data.shape
block_shape = (2, 2)

bsm = BlockSparseMatrix.BlockSparseMatrix(dense_data=dense_data,
                                          shape=shape,
                                          block_shape=block_shape)

block_mask = bsm.get_mask()
print('bsm:', bsm.dense_data)
print('block_mask:', block_mask)
