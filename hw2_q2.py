import math
import torch
import hw2_utils as utils
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

''' Translation of mjt-old's solutions to PyTorch '''
# Problem Gaussian Naive Bayes
def gaussian_theta(X, y):
    '''
    Arguments:
        X (S x N FloatTensor): features of each object
        y (S LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x N Float Tensor): MAP estimation of mu in N(mu, sigma2)

    '''
    X = X.float()
    mu = torch.stack([torch.mean(X[y == c], dim=0) for c in (0, 1)])
    sigma2 = torch.stack([torch.var(X[y == c], dim=0, unbiased=False) for c in (0, 1)])
    return mu, sigma2

def gaussian_p(y):
    '''
    Arguments:
        y (S LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    return torch.mean((y == 0).float())

def gaussian_classify(mu, sigma2, p, X):
    '''
    Arguments:
        mu (2 x N Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x N Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (S x N LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (S LongTensor): label of each object for classification, y[i] = 0/1
    
    '''
    X = X.float()
    p = torch.as_tensor(p, dtype=X.dtype)
    log_prior = torch.log(torch.stack([p, 1 - p]))
    diff = X.unsqueeze(1) - mu.unsqueeze(0)
    log_pdf = -0.5 * torch.log(2 * math.pi * sigma2).unsqueeze(0) - diff ** 2 / (2 * sigma2.unsqueeze(0))
    scores = torch.sum(log_pdf, dim=2) + log_prior
    return torch.argmax(scores, dim=1)

