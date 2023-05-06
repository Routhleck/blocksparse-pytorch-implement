from blocksparse.block_sparse_api import blocksparse_matmul
import torch
m = 32
k = 32
n = 32
device = 'cuda'
A = torch.randn((m,) + (k,), device=device)
B = torch.randn((k,) + (n,), device=device) # * (bm.random.rand(k, n) < 0.1)
A_temp = A
B_temp = B
C1 = blocksparse_matmul(A, B)
C_true = torch.matmul(A, B)
print(C1.cpu().numpy())
print(C_true.cpu().numpy())
print((C1.cpu().numpy() - C_true.cpu().numpy()))
# print(C3.numpy())
# print((C1 - C3).numpy())
