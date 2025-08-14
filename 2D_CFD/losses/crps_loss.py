import torch


def crps_loss(predictions, targets):
    batch_size, n_samples, n_vars, height, width = predictions.shape
    
    predictions_flat = predictions.reshape(batch_size, n_samples, -1)
    targets_flat = targets.reshape(batch_size, -1)
    
    targets_expanded = targets_flat.unsqueeze(1)
    
    first_term = torch.abs(predictions_flat - targets_expanded).mean(dim=1)
    
    predictions_expanded_1 = predictions_flat.unsqueeze(2)
    predictions_expanded_2 = predictions_flat.unsqueeze(1)
    
    pairwise_diff = torch.abs(predictions_expanded_1 - predictions_expanded_2)
    
    if n_samples > 1:
        second_term = pairwise_diff.sum(dim=(1, 2)) / (n_samples * (n_samples - 1))
    else:
        second_term = torch.zeros_like(first_term)
    
    crps = first_term - 0.5 * second_term
    
    loss = crps.mean()
    
    return loss
