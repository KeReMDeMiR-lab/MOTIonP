# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional

from .id_decoder import IDDecoder
from .motion_prediction import MotionPredictionHead, MotionPredictionLoss
from .trajectory_modeling import TrajectoryModeling


class MOTIP(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 256,
            num_queries: int = 300,
            num_classes: int = 1,
            motion_dim: int = 32,
            ffn_dim_ratio: int = 4,
            motion_weight: float = 0.35,
            **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.motion_weight = motion_weight

        # ID prediction components
        self.id_decoder = IDDecoder(hidden_dim, num_queries, num_classes, **kwargs)

        # Motion prediction components
        self.motion_head = MotionPredictionHead(hidden_dim, motion_dim, ffn_dim_ratio)
        self.motion_loss = MotionPredictionLoss()

        # Trajectory modeling
        self.trajectory_modeling = TrajectoryModeling(hidden_dim)

    def forward(self, 
                src: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                pos_embed: Optional[torch.Tensor] = None,
                query_embed: Optional[torch.Tensor] = None,
                trajectory_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MOTIP with motion prediction
        Args:
            src: [B, C, H, W] backbone features
            mask: [B, H, W] mask for padded regions
            pos_embed: [B, C, H, W] positional embeddings
            query_embed: [num_queries, C] query embeddings
            trajectory_features: [B, T, C] trajectory features
        Returns:
            Dict containing:
                - pred_logits: [B, num_queries, num_classes] class predictions
                - pred_boxes: [B, num_queries, 4] box predictions
                - pred_motion: [B, num_queries, 4] motion predictions
                - motion_conf: [B, num_queries, 1] motion confidence
        """
        # Get ID prediction outputs
        id_outputs = self.id_decoder(src, mask, pos_embed, query_embed)
        
        # Get trajectory features if not provided
        if trajectory_features is None:
            trajectory_features = self.trajectory_modeling(id_outputs['features'])
        
        # Predict motion
        motion_outputs = self.motion_head(trajectory_features)
        
        return {
            'pred_logits': id_outputs['pred_logits'],
            'pred_boxes': id_outputs['pred_boxes'],
            'pred_motion': motion_outputs['motion'],
            'motion_conf': motion_outputs['confidence'],
            'features': id_outputs['features']
        }
    
    def compute_loss(self,
                    outputs: Dict[str, torch.Tensor],
                    targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Compute combined ID and motion prediction losses
        Args:
            outputs: Model outputs from forward pass
            targets: List of target dictionaries
        Returns:
            Dict containing all loss components
        """
        # Compute ID prediction loss
        id_losses = self.id_decoder.compute_loss(outputs, targets)
        
        # Compute motion prediction loss
        motion_losses = self.motion_loss(
            outputs['pred_motion'],
            torch.cat([t['motion'] for t in targets], dim=0),
            outputs['pred_boxes'],
            torch.cat([t['boxes'] for t in targets], dim=0),
            weights=outputs['motion_conf'].squeeze(-1)
        )
        
        # Combine losses
        total_loss = id_losses['loss'] + self.motion_weight * sum(motion_losses.values())
        
        return {
            'loss': total_loss,
            'id_loss': id_losses['loss'],
            'motion_l1_loss': motion_losses['motion_l1_loss'],
            'motion_giou_loss': motion_losses['motion_giou_loss']
        }