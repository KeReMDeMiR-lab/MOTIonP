import torch
import torch.nn.functional as F
from typing import Optional, Dict

def compute_motion_targets(boxes: torch.Tensor, 
                         prev_boxes: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute motion targets from boxes
    Args:
        boxes: [N, 4] current boxes (x, y, w, h)
        prev_boxes: [N, 4] previous boxes (x, y, w, h)
    Returns:
        [N, 4] motion targets (dx, dy, dw, dh)
    """
    if prev_boxes is None:
        # If no previous boxes, assume no motion
        return torch.zeros_like(boxes)
    
    # Compute motion as difference between current and previous boxes
    motion = boxes - prev_boxes
    
    return motion

def smooth_motion_predictions(predictions: torch.Tensor,
                            window_size: int = 3) -> torch.Tensor:
    """
    Smooth motion predictions using a moving average
    Args:
        predictions: [T, N, 4] motion predictions
        window_size: size of the smoothing window
    Returns:
        [T, N, 4] smoothed predictions
    """
    if window_size <= 1:
        return predictions
    
    # Pad predictions for window
    pad = window_size // 2
    padded = F.pad(predictions, (0, 0, pad, pad), mode='replicate')
    
    # Create smoothing kernel
    kernel = torch.ones(1, 1, window_size, 1, device=predictions.device) / window_size
    
    # Apply smoothing to each dimension
    smoothed = []
    for i in range(4):
        dim_pred = padded[..., i:i+1].unsqueeze(0)
        smoothed_dim = F.conv2d(dim_pred, kernel, padding=(pad, 0))
        smoothed.append(smoothed_dim.squeeze(0))
    
    return torch.cat(smoothed, dim=-1)

def apply_motion_to_boxes(boxes: torch.Tensor,
                         motion: torch.Tensor) -> torch.Tensor:
    """
    Apply motion predictions to boxes
    Args:
        boxes: [N, 4] current boxes (x, y, w, h)
        motion: [N, 4] motion predictions (dx, dy, dw, dh)
    Returns:
        [N, 4] predicted boxes
    """
    return boxes + motion

def compute_motion_metrics(pred_motion: torch.Tensor,
                          target_motion: torch.Tensor,
                          pred_boxes: torch.Tensor,
                          target_boxes: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute motion prediction metrics
    Args:
        pred_motion: [N, 4] predicted motion
        target_motion: [N, 4] target motion
        pred_boxes: [N, 4] predicted boxes
        target_boxes: [N, 4] target boxes
    Returns:
        Dict containing motion metrics
    """
    # L1 error
    l1_error = F.l1_loss(pred_motion, target_motion, reduction='none').mean(dim=1)
    
    # Motion direction error (in degrees)
    pred_direction = torch.atan2(pred_motion[:, 1], pred_motion[:, 0])
    target_direction = torch.atan2(target_motion[:, 1], target_motion[:, 0])
    direction_error = torch.abs(pred_direction - target_direction) * 180 / torch.pi
    
    # Speed error
    pred_speed = torch.norm(pred_motion[:, :2], dim=1)
    target_speed = torch.norm(target_motion[:, :2], dim=1)
    speed_error = torch.abs(pred_speed - target_speed)
    
    return {
        'l1_error': l1_error,
        'direction_error': direction_error,
        'speed_error': speed_error
    } 