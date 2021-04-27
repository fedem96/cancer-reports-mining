from scipy.sparse import csr_matrix
import torch


def sparse_tensor_to_csr_matrix(tensor):
    assert tensor.ndim == 2 or (tensor.ndim == 3 and tensor.shape[1] == 1)   # invalid tensor shape
    if tensor.ndim == 3:
        tensor = torch.sparse.sum(tensor, 1)
    return csr_matrix((tensor.values().numpy(), tensor.indices().numpy()), shape=tensor.shape)
