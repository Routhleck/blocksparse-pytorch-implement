from blocksparse.block_sparse_api import blocksparse_matmul
import torch
m = 64
k = 32
n = 64
def gpu_test(device = 'cuda'):
    A = torch.randn((m,) + (k,), device=device)
    B = torch.randn((k,) + (n,), device=device) # * (bm.random.rand(k, n) < 0.1)
    A_temp = A
    B_temp = B
    C1 = blocksparse_matmul(A, B, device=device)
    C_true = torch.matmul(A, B)
    print(C1.cpu().numpy())
    print(C_true.cpu().numpy())
    print((C1.cpu().numpy() - C_true.cpu().numpy()))
    # print(C3.numpy())
    # print((C1 - C3).numpy())

def cpu_test(device = 'cpu'):
    A = torch.randn((m,) + (k,), device=device)
    B = torch.randn((k,) + (n,), device=device)
    A_temp = A
    B_temp = B
    C1 = blocksparse_matmul(A, B)
    C_true = torch.matmul(A, B)
    print(C1.cpu().numpy())
    print(C_true.cpu().numpy())
    print((C1.cpu().numpy() - C_true.cpu().numpy()))

gpu_test()
cpu_test()
