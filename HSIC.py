#The following is the implementation of the HSIC loss. The code is based on the implementation of Atzmon et al. and adapted to our needs, for the causaltriplet settings.

import torch
import numpy as np
import pdb

def pairwise_distances(x):
  """
    Function to compute the pairwise distance matrix in a batch. 
    Args:
        x: pytorch tensor of shape (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: pytorch tensor of shape (batch_size, num_points, num_points)
  """

  x_distances = torch.sum(x**2,-1).reshape((-1,1))
  return -2*torch.mm(x,x.t()) + x_distances + x_distances.t() 

def kernelMatrixGaussian(x, sigma=1):
    """
      Function to compute the Gaussian kernel matrix in a batch.
      Args:
          x: pytorch tensor of shape (batch_size, num_points, num_dims)
          sigma: bandwidth of the Gaussian kernel
      Returns:
          gaussian kernel matrix: pytorch tensor of shape (batch_size, num_points, num_points)

    """

    pairwise_distances_ = pairwise_distances(x)
    gamma = -1.0 / (sigma ** 2)
    return torch.exp(gamma * pairwise_distances_)

def kernelMatrixLinear(x):
    """
      Function to compute the linear kernel matrix in a batch.
      Args:
          x: pytorch tensor of shape (batch_size, num_points, num_dims)
      Returns:
          linear kernel matrix: pytorch tensor of shape (batch_size, num_points, num_points)
    """
    return torch.matmul(x,x.t())


def HSIC(X, Y, kernelX="Gaussian", kernelY="Gaussian", sigmaX=1, sigmaY=1,
         log_median_pairwise_distance=False):
  """
    Function to compute the HSIC value between two matrices.
    Args:
        X: pytorch tensor of shape (num_points, num_dims)
        Y: pytorch tensor of shape (num_points, num_dims)
        kernelX: kernel type for X
        kernelY: kernel type for Y
        sigmaX: bandwidth of the Gaussian kernel for X
        sigmaY: bandwidth of the Gaussian kernel for Y
        log_median_pairwise_distance: whether to log the median pairwise distance
    Returns:
        HSIC value: pytorch tensor of shape (1)
        median pairwise distance of X: pytorch tensor of shape (1)
        median pairwise distance of Y: pytorch tensor of shape (1)
  """

  m,_ = X.shape
  assert(m>1)

  
  K = kernelMatrixGaussian(X,sigmaX) if kernelX == "Gaussian" else kernelMatrixLinear(X)
  L = kernelMatrixGaussian(Y,sigmaY) if kernelY == "Gaussian" else kernelMatrixLinear(Y)

  
  K=K.float().cuda()

  H = torch.eye(m, device='cuda') - 1.0/m * torch.ones((m,m), device='cuda')
  H = H.float().cuda()

  Kc = torch.mm(H,torch.mm(K,H))

  HSIC = torch.trace(torch.mm(L,Kc))/((m-1)**2)
  return HSIC




def conditional_indep_losses(repr1, repr2, y, indep_coeff, num_classes_action=None,
                             Hkernel='L', Hkernel_sigma_obj=None, Hkernel_sigma_attr=None,
                              device=None):
    """
    Function to compute the conditional independence loss between two representations.
    Args:
        repr1: pytorch tensor of shape (num_points, num_dims) - repr1, repr2 are phi_hat1, phi_hat2 in the paper from Atzmon et al.
        repr2: pytorch tensor of shape (num_points, num_dims)
        y: pytorch tensor of shape (num_points, num_dims) - these are the action labels (ground truth)
        indep_coeff: coefficient for the conditional independence loss
        num_classes_action: number of action labels
        Hkernel: kernel type for the HSIC loss
        Hkernel_sigma_obj: bandwidth of the Gaussian kernel for the HSIC loss
        Hkernel_sigma_attr: bandwidth of the Gaussian kernel for the HSIC loss
        log_median_pairwise_distance: whether to log the median pairwise distance
        device: device to run the code on

    Returns:
        conditional independence loss: pytorch tensor of shape (1)
    """
    

    #We are working in the case we are only predicting the action.

    HSIC_tmp_loss = 0.
    
    labels_in_batch_sorted, indices = torch.sort(y)
    #nonzero returns a list of indices of the elements different from 0
    unique_ixs = 1 + torch.nonzero(labels_in_batch_sorted[1:] - labels_in_batch_sorted[:-1], as_tuple=False)
    unique_ixs = [0] + unique_ixs.flatten().cpu().numpy().tolist() + [len(y)]
    
    for j in range(len(unique_ixs)-1):
        current_class_indices = unique_ixs[j], unique_ixs[j + 1]
        count = current_class_indices[1] - current_class_indices[0]
        if count < 2:
            continue
        curr_class_slice = slice(*current_class_indices)
        curr_class_indices = indices[curr_class_slice].sort()[0]

        HSIC_kernel = dict(G='Gaussian', L='Linear')[Hkernel]

        hsic_loss_i = \
          HSIC(repr1[curr_class_indices, :].float(), repr2[curr_class_indices, :].float(),
          kernelX=HSIC_kernel, kernelY=HSIC_kernel,
          sigmaX=Hkernel_sigma_obj, sigmaY=Hkernel_sigma_attr)
        

        HSIC_tmp_loss += hsic_loss_i
        

    HSIC_tmp_loss = HSIC_tmp_loss / num_classes_action

    indep_loss = torch.tensor(0.).to(device)
    indep_loss = indep_coeff * HSIC_tmp_loss
    return indep_loss