import torch
import torch.nn as nn
import torch.nn.functional as F

class IDLoss(nn.Module):
    """
    Index Diversion (ID) Loss based on Chai et al. (CVPR 2023).
    Instead of mimicking a teacher, it pushes features away from the 
    'Unrecognizable Plates' (UPs) cluster center.
    """
    def __init__(self, margin=0.5, eps=1e-8, **kwargs):
        super(IDLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, student_feat, labels=None):
        """
        Args:
            student_feat: [B, C, H, W] or [B, L, C]
        """
        if len(student_feat.shape) == 4:
            # Flatten spatial dims to get embedding vectors
            # [B, C, H, W] -> [B, C] via Global Average Pooling
            feat = F.adaptive_avg_pool2d(student_feat, (1, 1)).flatten(1)
        else:
            # [B, L, C] -> [B, C]
            feat = student_feat.mean(dim=1)

        # Normalize features to hypersphere
        feat = F.normalize(feat, p=2, dim=1)

        # Calculate the center of 'Unrecognizable' features in the batch.
        # As a proxy, we use the batch mean (global noise center).
        # In a more advanced version, we could weight samples by their CTC confidence.
        center = feat.mean(dim=0, keepdim=True)
        center = F.normalize(center, p=2, dim=1)

        # Distance to the UI center (Cosine Similarity)
        # Similarity = 1 means feature is exactly at the noise center
        sim = torch.matmul(feat, center.t()) # [B, 1]

        # ID Loss: Minimize similarity to the noise center
        # We use an exponential penalty or a hinge loss
        loss = torch.mean(torch.exp(sim - self.margin))

        return loss
