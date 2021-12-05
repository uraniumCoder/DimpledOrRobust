import torch 
"""Code to do projection of vector onto a linear subspace"""
def projection(b, spanning_vector_matrix):
    #spanning vector matrix is of shape p,m
    #b is of shape n,1. Here p<n We stack zeros under A so that it becomes a matrix of shape n,m.
    b = b.type(torch.FloatTensor)
    n = b.shape[0]
    (p,m)= spanning_vector_matrix.shape
    zero_matrix = torch.zeros(n-p,m)
    A = torch.cat((spanning_vector_matrix,zero_matrix),dim=0)
    A_T = torch.transpose(A,0,1)
    inv = torch.linalg.inv(torch.matmul(A_T,A))
    mat1 = torch.matmul(A,inv)
    mat2 = torch.matmul(mat1,A_T)

    return torch.matmul(mat2,b)