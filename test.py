import blocksparse.BlockSparseMatrix as BlockSparseMatrix

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sizes = [640, 320, 960]
block_shape = (32, 32)
density = 0.1
device = 'cuda'
block_count = None

a = torch.randn((sizes[0], ) + (sizes[1], ), device=device).abs()

total_block_count = sizes[1] * sizes[2] / block_shape[0] / block_shape[1]
block_count = int(total_block_count * density)

bsm = BlockSparseMatrix.BlockSparseMatrix.randn((sizes[1],sizes[2]), n_blocks=block_count, block_shape=block_shape)

c = bsm.matmul(a, False)

# verbose
print('a:', a)
print('b:', bsm.data)
print('c:', c)

import pandas as pd
import numpy as np
a = a.cpu().numpy()
b = bsm.to_dense().cpu().detach().numpy()
b_data = bsm.data.cpu().detach().numpy()
c = c.cpu().numpy()
a = pd.DataFrame(a)
b = pd.DataFrame(b)
b_data = pd.DataFrame(b_data)
c = pd.DataFrame(c)

# 创建test_out文件夹
if not os.path.exists('test_out'):
    os.makedirs('test_out')

a.to_csv('test_out/a.csv', index=False, header=False)
b.to_csv('test_out/b.csv', index=False, header=False)
b_data.to_csv('test_out/b_data.csv', index=False, header=False)
c.to_csv('test_out/c.csv', index=False, header=False)

# 将a,b,c绘制成热力图
import matplotlib.pyplot as plt
import seaborn as sns

a = pd.read_csv('test_out/a.csv', header=None)
b = pd.read_csv('test_out/b.csv', header=None)
b_data = pd.read_csv('test_out/b_data.csv', header=None)
c = pd.read_csv('test_out/c.csv', header=None)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.heatmap(a, ax=axes[0, 0], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(b, ax=axes[0, 1], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(b_data, ax=axes[1, 0], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(c, ax=axes[1, 1], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)

# 设置标题
axes[0, 0].set_title('a')
axes[0, 1].set_title('b_dense')
axes[1, 0].set_title('b_data')
axes[1, 1].set_title('c')
# 设置主标题
fig.suptitle('size: %s, block_size: %s, density: %.2f' % (str(sizes), str(block_shape), density), fontsize=20)
# 保存图片
plt.savefig('test_out/a_b_c.png', dpi=400, bbox_inches='tight')
plt.show()