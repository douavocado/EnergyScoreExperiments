import torch


def energy_score_loss(predictions, targets, beta=1.0):
    batch_size, n_samples, n_vars, height, width = predictions.shape
    
    predictions_flat = predictions.reshape(batch_size, n_samples, -1)
    targets_flat = targets.reshape(batch_size, -1)
    
    targets_expanded = targets_flat.unsqueeze(1)
    
    diff_to_target = predictions_flat - targets_expanded
    if beta == 1.0:
        norm_to_target = torch.abs(diff_to_target).sum(dim=-1)
    elif beta == 2.0:
        norm_to_target = torch.norm(diff_to_target, p=2, dim=-1)
    else:
        norm_to_target = torch.norm(diff_to_target, p=beta, dim=-1)
    
    first_term = norm_to_target.mean(dim=1)
    
    if n_samples > 1:
        predictions_expanded_1 = predictions_flat.unsqueeze(2)
        predictions_expanded_2 = predictions_flat.unsqueeze(1)
        
        pairwise_diff = predictions_expanded_1 - predictions_expanded_2
        
        if beta == 1.0:
            pairwise_norms = torch.abs(pairwise_diff).sum(dim=-1)
        elif beta == 2.0:
            pairwise_norms = torch.norm(pairwise_diff, p=2, dim=-1)
        else:
            pairwise_norms = torch.norm(pairwise_diff, p=beta, dim=-1)
        
        mask = ~torch.eye(n_samples, dtype=torch.bool, device=predictions.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        masked_norms = pairwise_norms * mask.float()
        second_term = masked_norms.sum(dim=(1, 2)) / (n_samples * (n_samples - 1))
    else:
        second_term = torch.zeros_like(first_term)
    
    energy_score = first_term - 0.5 * second_term
    
    loss = energy_score.mean()
    
    return loss
