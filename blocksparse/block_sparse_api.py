import torch


def get_mask(dense_b, blockshape, blockcount):
    mask = torch.zeros(blockcount[1] * blockcount[0], dtype=torch.bool, device=dense_b.device)

    for i in range(blockcount[1]):
        for j in range(blockcount[0]):
            # if data exist, set related mask to True
            if dense_b[i * blockshape[1]: (i + 1) * blockshape[1],
               j * blockshape[0]: (j + 1) * blockshape[0]].sum() != 0:
                mask[i * blockcount[0] + j] = True

    # reshape
    mask = mask.reshape(blockcount[1], blockcount[0])
    return mask


def get_data(dense_b, mask, blockshape, blockcount, n_blocks):
    data = torch.zeros(
        (n_blocks * blockshape[1], blockshape[0]),
        dtype=torch.float,
        device=dense_b.device,
    )

    assignment_count = 0
    for i in range(blockcount[1]):
        for j in range(blockcount[0]):
            if mask[i, j]:
                data[assignment_count * blockshape[1]: (assignment_count + 1) * blockshape[1],
                :] = dense_b[i * blockshape[1]: (i + 1) * blockshape[1],
                     j * blockshape[0]: (j + 1) * blockshape[0]]
                assignment_count += 1
    return data


def get_ptr_indices(mask, blockcount, n_blocks, block_ptr=None):
    nnz = torch.nonzero(mask, as_tuple=False)

    if block_ptr is None:
        block_ptr = torch.arange(0, nnz.shape[0], device=mask.device)

    _, indices = block_ptr.sort()
    blocks = nnz[indices]

    blocks = blocks.flip(-1).flatten().to(dtype=torch.int32)

    nnzt = nnz.transpose(0, 1)
    X, Y = blockcount[1], blockcount[0]
    rows = nnzt[0]
    cols = nnzt[1]

    block_indices = torch.zeros(X * Y, dtype=torch.long, device=mask.device)
    positions = rows * Y + cols
    block_indices[positions] = block_ptr + 1
    block_indices = block_indices.reshape(X, Y).t().reshape(X * Y)
    block_ptr = block_indices[torch.nonzero(block_indices, as_tuple=False)] - 1
    block_ptr = block_ptr.squeeze(-1)

    X, Y = Y, X
    rows = cols
    nnztt = torch.nonzero(mask.t(), as_tuple=False)
    cols = nnztt[:, 1]

    ptr_b = torch.zeros((X + 1), dtype=torch.long, device=mask.device)
    ptr_b.index_add_(
        0,
        rows + 1,
        torch.ones(size=(cols.shape[0],), dtype=torch.long, device=mask.device),
    )

    ptr_b = ptr_b.cumsum(0).to(dtype=torch.int32)
    indices_b = torch.stack([cols, block_ptr], 1).to(dtype=torch.int32)

    return ptr_b, indices_b

def blocksparse_matmat_cpu_(
            A_data,
            B_ptr,
            B_indices,
            B_data,
            C_data,
            m,
            n,
            k,
            block_size_rows_b,
            block_size_cols_b):
    res_val = torch.zeros((m, n))
    # index[0]为index, index[1]为该index对应的block的index
    for idx, index in enumerate(B_indices):
        # find the column
        row_index = 0
        for ptr in B_ptr[1:]:
            if ptr > idx:
                break
            row_index += 1
        row_start = row_index * block_size_cols_b
        # find the row
        col_start = index[0] * block_size_rows_b
        # calculate the value and add to the res_val
        for i in range(block_size_rows_b):
            for j in range(block_size_cols_b):
                if B_data[index[1] * block_size_rows_b + i, j] == 0:
                    continue
                row_now = row_start + j
                res_val[:, row_now] += A_data[:, col_start + i] * B_data[index[1] * block_size_rows_b + i, j]
                '''for c_m in range(m):
                    print('c{c_col}{c_row} = a{a_col}{a_row} * b{b_col}{b_row}'.format(c_col=c_m + 1,c_row=row_now + 1,
                                                                                     a_col=c_m + 1, a_row=col_start + i + 1,
                                                                                     b_col=col_start + i + 1, b_row=row_start + j + 1))
                    res_val[c_m, row_now] += A_data[c_m, col_start + i] * B_data[index[1] * block_size_rows_b + i, j]'''

    return res_val

def blocksparse_matmul(dense_a, dense_b, blockshape=(32, 32), device='cpu'):
    # m, n, k
    m = dense_a.shape[0]
    n = dense_b.shape[1]
    k = dense_a.shape[1]

    # blockcount
    blockcount = (n // blockshape[0], k // blockshape[1])

    # mask
    mask = get_mask(dense_b, blockshape, blockcount)

    # n_blocks
    n_blocks = mask.sum()

    # data_b
    data_b = get_data(dense_b, mask, blockshape, blockcount, n_blocks)

    # ptr_b, indices_b
    ptr_b, indices_b = get_ptr_indices(mask, blockcount, n_blocks)

    # out
    out = torch.zeros((n, m), device=dense_a.device)

    # dense_a = dense_a.contiguous()
    # ptr_b = ptr_b.contiguous()
    # indices_b = indices_b.contiguous()
    # data_b = data_b.contiguous()

    if device == 'cpu':
        out = blocksparse_matmat_cpu_(dense_a,
        ptr_b,
        indices_b,
        data_b,
        m,
        n,
        k,
        blockshape[0],
        blockshape[1],
        out)
        return out
    elif device == 'gpu' or device == 'cuda':
        import block_sparse_native
        block_sparse_native.blocksparse_matmul_cutlass(
            dense_a,
            True,
            ptr_b,
            indices_b,
            data_b,
            m,
            n,
            k,
            blockshape[0],
            blockshape[1],
            out,
        )
        return out.t()
    else:
        raise Exception('Invalid device: ', device)



