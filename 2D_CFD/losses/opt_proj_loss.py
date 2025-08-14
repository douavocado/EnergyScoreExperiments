import math
import torch
import torch.nn as nn


def opt_proj_loss(predictions, targets, step_size=0.2, min_iterations=1, eps=1e-12):
    batch_size, n_samples, n_vars, height, width = predictions.shape
    
    predictions_flat = predictions.reshape(batch_size, n_samples, -1)
    targets_flat = targets.reshape(batch_size, -1)
    
    D = predictions_flat.shape[2]
    device = predictions.device
    dtype = predictions.dtype
    
    batch_losses = []
    
    for b in range(batch_size):
        X = predictions_flat[b]
        y = targets_flat[b]
        
        theta = _compute_theta(X, y, step_size, min_iterations, eps)
        
        pred_proj = X @ theta
        true_proj = y @ theta
        
        crps_1d = _crps_1d(pred_proj, true_proj)
        batch_losses.append(crps_1d)
    
    loss = torch.stack(batch_losses).mean()
    return loss


def _normalise_vector(v, eps):
    norm = torch.norm(v)
    if norm <= eps:
        out = torch.zeros_like(v)
        out[0] = 1.0
        return out
    return v / norm


def _compute_theta(X, y, step_size, min_iterations, eps):
    N, D = X.shape
    device = X.device
    dtype = X.dtype
    
    mean_x = X.mean(dim=0)
    theta = y - mean_x
    theta = _normalise_vector(theta, eps)
    
    K = max(min_iterations, int(math.log(max(N, 2))))
    
    X_detached = X.detach()
    y_detached = y.detach()
    
    with torch.no_grad():
        for _ in range(K):
            x_theta = X_detached @ theta
            y_theta = torch.dot(y_detached, theta)
            
            E_X = (X_detached.t() @ x_theta) / N
            E_Y = y_detached * y_theta
            
            S_X = (x_theta.pow(2).sum()) / N
            S_Y = y_theta.pow(2)
            
            Sigma_theta = 0.5 * (E_X + E_Y)
            DeltaSigma_theta = 0.5 * (E_Y - E_X)
            
            b = 0.5 * (S_X + S_Y)
            a = 0.5 * (S_Y - S_X)
            
            denom = torch.clamp(b, min=eps).pow(2.5)
            
            grad = (4.0 * a * b) * DeltaSigma_theta - (3.0 * a * a) * Sigma_theta
            grad = grad / denom
            
            theta_dot_grad = torch.dot(theta, grad)
            g = grad - theta_dot_grad * theta
            
            theta = theta + step_size * g
            theta = _normalise_vector(theta, eps)
    
    return theta.detach()


def _crps_1d(pred_samples, true_value):
    N = pred_samples.shape[0]
    
    first_term = torch.abs(pred_samples - true_value).mean()
    
    if N > 1:
        pred_expanded_1 = pred_samples.unsqueeze(1)
        pred_expanded_2 = pred_samples.unsqueeze(0)
        pairwise_diff = torch.abs(pred_expanded_1 - pred_expanded_2)
        
        mask = ~torch.eye(N, dtype=torch.bool, device=pred_samples.device)
        second_term = (pairwise_diff * mask.float()).sum() / (N * (N - 1))
    else:
        second_term = torch.tensor(0.0, device=pred_samples.device, dtype=pred_samples.dtype)
    
    crps = first_term - 0.5 * second_term
    return crps
