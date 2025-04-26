# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import torch.nn as nn

from models.ffn import FFN


class TrajectoryModeling(nn.Module):
    def __init__(
            self,
            detr_dim: int,
            ffn_dim_ratio: int,
            feature_dim: int,
            motion_dim: int = 32,
    ):
        super().__init__()

        self.detr_dim = detr_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim

        # Appearance feature adapter
        self.adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        
        # Motion feature extraction components
        self.motion_encoder = nn.Sequential(
            nn.Linear(12, motion_dim // 2),  # 4 for position, 4 for velocity, 4 for acceleration
            nn.GELU(),
            nn.Linear(motion_dim // 2, motion_dim)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + motion_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        pass
    
    def compute_motion_features(self, trajectory_boxes, trajectory_masks, trajectory_times):
        """
        Compute motion features based on position, velocity, and acceleration
        """
        B, G, T, N, _ = trajectory_boxes.shape
        motion_features = torch.zeros(B, G, T, N, self.motion_dim, dtype=trajectory_boxes.dtype, device=trajectory_boxes.device)
        
        # Compute for each batch and group
        for b in range(B):
            for g in range(G):
                # Process each object trajectory
                for n in range(N):
                    # Skip if all frames are masked
                    if trajectory_masks[b, g, :, n].all():
                        continue
                    
                    # Get valid frames for this trajectory
                    valid_mask = ~trajectory_masks[b, g, :, n]
                    if valid_mask.sum() < 2:  # Need at least 2 frames for velocity
                        continue
                        
                    valid_boxes = trajectory_boxes[b, g, valid_mask, n]
                    valid_times = trajectory_times[b, g, valid_mask, n]
                    
                    # Sort by time to ensure correct computation
                    time_indices = torch.argsort(valid_times)
                    valid_boxes = valid_boxes[time_indices]
                    valid_times = valid_times[time_indices]
                    
                    # Compute velocities
                    velocities = torch.zeros_like(valid_boxes)
                    for i in range(1, len(valid_boxes)):
                        time_diff = max((valid_times[i] - valid_times[i-1]).item(), 1)
                        velocities[i] = (valid_boxes[i] - valid_boxes[i-1]) / time_diff
                    
                    # Compute accelerations
                    accelerations = torch.zeros_like(valid_boxes)
                    for i in range(2, len(valid_boxes)):
                        time_diff = max((valid_times[i] - valid_times[i-1]).item(), 1)
                        accelerations[i] = (velocities[i] - velocities[i-1]) / time_diff
                    
                    # Create motion features for each valid frame
                    for i, time_idx in enumerate(valid_times):
                        frame_idx = time_idx.item()
                        if frame_idx < T:
                            pos = valid_boxes[i]
                            vel = velocities[i] if i > 0 else torch.zeros_like(valid_boxes[i])
                            acc = accelerations[i] if i > 1 else torch.zeros_like(valid_boxes[i])
                            
                            # Concatenate position, velocity, and acceleration
                            motion_vector = torch.cat([pos, vel, acc], dim=0)
                            
                            # Encode motion into feature vector
                            motion_features[b, g, frame_idx, n] = self.motion_encoder(motion_vector)
        
        return motion_features

    def forward(self, seq_info):
        # Process appearance features
        trajectory_features = seq_info["trajectory_features"]
        trajectory_features = trajectory_features + self.adapter(trajectory_features)
        trajectory_features = self.norm(trajectory_features)
        
        # Compute motion features
        motion_features = self.compute_motion_features(
            seq_info["trajectory_boxes"],
            seq_info["trajectory_masks"],
            seq_info["trajectory_times"]
        )
        
        # Fuse appearance and motion features
        fused_features = self.fusion(
            torch.cat([trajectory_features, motion_features], dim=-1)
        )
        
        # Apply final FFN
        fused_features = fused_features + self.ffn(fused_features)
        fused_features = self.ffn_norm(fused_features)
        
        seq_info["trajectory_features"] = fused_features
        seq_info["motion_features"] = motion_features
        
        return seq_info