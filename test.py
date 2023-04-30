import torch

from blocksparse.BlockSparseMatrix import BlockSparseMatrix_torch
import pandas as pd

sizes = [640, 320, 960]
block_size = [32, 32]
density = 0.5
blocks = None # [(0, 0)]
device = 'cuda'
block_count = None

# a = torch.randn((sizes[0], ) + (sizes[1], ), device=device).abs()
a = torch.tensor(pd.read_csv('test_input/a.csv', header=None).values, device=device)

if block_count is None and blocks is None:
    total_block_count = sizes[1] * sizes[2] / block_size[0] / block_size[1]
    block_count = int(total_block_count * density)

# bsm = BlockSparseMatrix_torch.randn(
#             (sizes[2], sizes[1]),
#             block_count,
#             # blocks=blocks,
#             block_shape=block_size,
#             device=device,
#             positive=True,)
bsm = BlockSparseMatrix_torch.from_dense(torch.tensor(pd.read_csv('test_input/b.csv', header=None).values, device=device), block_shape=block_size)

c = bsm.matmul(a, transpose=True)
a = a.float()
c_true_pytorch = a.matmul(bsm.to_dense().transpose(0, 1))

print('a:', a)
print('b:', bsm.data)
print('c:', c)

# print(c == c_true_pytorch)

# 将a,b,c存储到csv中, a,b,c的shape都是(64, 64)
import pandas as pd

a = a.cpu().numpy()
b = bsm.to_dense().cpu().detach().numpy()
b_data = bsm.data.cpu().detach().numpy()
c = c.cpu().detach().numpy()
c_true_pytorch = c_true_pytorch.cpu().detach().numpy()
a = pd.DataFrame(a)
b = pd.DataFrame(b)
b_data = pd.DataFrame(b_data)
c = pd.DataFrame(c)
c_true_pytorch = pd.DataFrame(c_true_pytorch)
a.to_csv('test_out/a.csv', index=False, header=False)
b.to_csv('test_out/b.csv', index=False, header=False)
b_data.to_csv('test_out/b_data.csv', index=False, header=False)
c.to_csv('test_out/c.csv', index=False, header=False)
c_true_pytorch.to_csv('test_out/c_true.csv', index=False, header=False)

# 将a,b,c绘制成热力图
import matplotlib.pyplot as plt
import seaborn as sns

a = pd.read_csv('test_out/a.csv', header=None)
b = pd.read_csv('test_out/b.csv', header=None)
c_blocksparse_pytorch = pd.read_csv('test_out/c.csv', header=None)
c_blocksparse = pd.read_csv('test_input/c_blocksparse.csv', header=None)
c_true_pytorch = pd.read_csv('test_out/c_true.csv', header=None)
c_true_jnp = pd.read_csv('test_input/c_true_jnp.csv', header=None)


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
sns.heatmap(a, ax=axes[0, 0], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(b, ax=axes[0, 1], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(c_blocksparse_pytorch, ax=axes[1, 0], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)
# sns.heatmap(c_blocksparse, ax=axes[1, 0], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(c_true_pytorch, ax=axes[1, 1], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)
# sns.heatmap(c_true_jnp, ax=axes[1, 2], cmap='Blues', square=True, cbar=False, xticklabels=False, yticklabels=False)


# 设置标题
axes[0, 0].set_title('a' + str(a.shape))
axes[0, 1].set_title('b_dense' + str(b.shape))
axes[1, 0].set_title('c_blocksparse_pytorch' + str(c_blocksparse_pytorch.shape))
#axes[1, 0].set_title('c_blocksparse' + str(c_blocksparse.shape))
axes[1, 1].set_title('c_true_pytorch' + str(c_true_pytorch.shape))
# axes[1, 2].set_title('c_true_jnp' + str(c_true_jnp.shape))


# 设置主标题
fig.suptitle('size: %s, block_size: %s, density: %.2f' % (str(sizes), str(block_size), density), fontsize=20)
# 保存图片
plt.savefig('test_out/a_b_c.png', dpi=400, bbox_inches='tight')
plt.show()


