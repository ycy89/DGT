import torch

def compute_centered_matrix(D):
    """计算中心化矩阵"""
    row_mean = torch.mean(D, dim=1, keepdim=True)
    col_mean = torch.mean(D, dim=0, keepdim=True)
    overall_mean = torch.mean(D)

    D_centered = D - row_mean - col_mean + overall_mean
    return D_centered


def distance_covariance(X, Y):
    """计算距离协方差"""
    D_X = torch.cdist(X, X, p=2)
    D_Y = torch.cdist(Y, Y, p=2)

    D_X_centered = compute_centered_matrix(D_X)
    D_Y_centered = compute_centered_matrix(D_Y)

    dCov = torch.sqrt(torch.mean(D_X_centered * D_Y_centered))
    return dCov


# 示例数据
X = torch.rand(100, 50)
Y = torch.rand(100, 50)

print('Distance Covariance:', distance_covariance(X, Y).item())
