import torch 
"""Code to do projection of vector onto a linear subspace"""
def projection(b, spanning_vector_matrix):
    #spanning vector matrix is of shape p,m
    #b is of shape n,1. Here p<n We stack zeros under A so that it becomes a matrix of shape n,m.

    b = b.type(torch.cuda.FloatTensor)
    n = b.shape[0]
    (p,m)= spanning_vector_matrix.shape
    zero_matrix = torch.zeros(n-p,m, device=b.device)
    A = torch.cat((spanning_vector_matrix,zero_matrix),dim=0)

    #A is of shape n,m
    """
    A_T = torch.transpose(A,0,1)
    inv = torch.linalg.inv(torch.matmul(A_T,A))
    mat1 = torch.matmul(A,inv)
    mat2 = torch.matmul(mat1,A_T)
    """
    coeffs = torch.lstsq(b, A)[0][:m] #change to torch.linalg.lstsq for torch version >= 1.1
    return A @ coeffs
