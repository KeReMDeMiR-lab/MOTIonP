import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

class MotionPredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, motion_dim: int = 32, ffn_dim_ratio: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.motion_dim = motion_dim
        self.ffn_dim = hidden_dim * ffn_dim_ratio
        
        # Motion prediction layers
        self.motion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, motion_dim * 4)  # Predict dx, dy, dw, dh
        )
        
        # Motion confidence prediction
        self.motion_conf = nn.Sequential(
            nn.Linear(hidden_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict motion parameters for each object
        Args:
            features: [N, hidden_dim] feature vectors
        Returns:
            Dict containing:
                - motion: [N, 4] predicted motion parameters (dx, dy, dw, dh)
                - confidence: [N, 1] motion prediction confidence
        """
        motion = self.motion_ffn(features)
        confidence = self.motion_conf(features)
        
        return {
            'motion': motion.view(-1, 4),
            'confidence': confidence
        }

class MotionPredictionLoss(nn.Module):
    def __init__(self, l1_weight: float = 5.0, giou_weight: float = 2.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        
    def forward(self, 
                pred_motion: torch.Tensor,
                target_motion: torch.Tensor,
                pred_boxes: torch.Tensor,
                target_boxes: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute motion prediction loss
        Args:
            pred_motion: [N, 4] predicted motion parameters
            target_motion: [N, 4] target motion parameters
            pred_boxes: [N, 4] predicted boxes (x, y, w, h)
            target_boxes: [N, 4] target boxes (x, y, w, h)
            weights: [N] optional weights for each prediction
        Returns:
            Dict containing loss components
        """
        # L1 loss for motion parameters
        l1_loss = F.l1_loss(pred_motion, target_motion, reduction='none')
        if weights is not None:
            l1_loss = l1_loss * weights.unsqueeze(-1)
        l1_loss = l1_loss.mean()
        
        # GIoU loss for predicted boxes
        pred_boxes_moved = pred_boxes + pred_motion
        giou_loss = 1 - self.giou(pred_boxes_moved, target_boxes)
        if weights is not None:
            giou_loss = giou_loss * weights
        giou_loss = giou_loss.mean()
        
        return {
            'motion_l1_loss': l1_loss * self.l1_weight,
            'motion_giou_loss': giou_loss * self.giou_weight
        }
    
    @staticmethod
    def giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute GIoU between two sets of boxes
        Args:
            boxes1: [N, 4] (x, y, w, h)
            boxes2: [N, 4] (x, y, w, h)
        Returns:
            [N] GIoU values
        """
        # Convert to (x1, y1, x2, y2) format
        boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] / 2,
                           boxes1[..., :2] + boxes1[..., 2:] / 2], dim=-1)
        boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] / 2,
                           boxes2[..., :2] + boxes2[..., 2:] / 2], dim=-1)
        
        # Compute intersection
        lt = torch.max(boxes1[..., :2], boxes2[..., :2])
        rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        
        # Compute union
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        union = area1 + area2 - inter
        
        # Compute IoU
        iou = inter / (union + 1e-7)
        
        # Compute enclosing box
        enclose_lt = torch.min(boxes1[..., :2], boxes2[..., :2])
        enclose_rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        
        # Compute GIoU
        giou = iou - (enclose_area - union) / (enclose_area + 1e-7)
        
        return giou 