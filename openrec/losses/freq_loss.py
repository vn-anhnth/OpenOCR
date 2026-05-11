import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqLoss(nn.Module):
    """
    Frequency-domain Knowledge Distillation Loss.
    Encourages the student to reconstruct high-frequency details (edges) 
    from the teacher's feature maps.
    """
    def __init__(self, hp_ratio=0.1, **kwargs):
        super(FreqLoss, self).__init__()
        self.hp_ratio = hp_ratio
        self.projs = nn.ModuleDict()

    def forward(self, student_feat, teacher_feat):
        # 0. Align channels if needed
        if student_feat.shape[1] != teacher_feat.shape[1]:
            key = f"{student_feat.shape[1]}_{teacher_feat.shape[1]}"
            if key not in self.projs:
                # Create a 1x1 conv to project student channels to teacher channels
                proj = nn.Conv2d(student_feat.shape[1], teacher_feat.shape[1], kernel_size=1).to(student_feat.device)
                self.projs[key] = proj
            student_feat = self.projs[key](student_feat)

        # Ensure same shape for KD
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            teacher_feat = F.interpolate(teacher_feat, size=student_feat.shape[2:], mode='bilinear', align_corners=False)

        # 1. FFT
        # Cast to float32 because fft2 doesn't support bfloat16/half
        s_fft = torch.fft.fft2(student_feat.float())
        t_fft = torch.fft.fft2(teacher_feat.float())
        
        # 2. Shift zero frequency to center
        s_fft_shift = torch.fft.fftshift(s_fft)
        t_fft_shift = torch.fft.fftshift(t_fft)
        
        # 3. High-pass Filter (Masking center low frequencies)
        B, C, H, W = student_feat.shape
        cy, cx = H // 2, W // 2
        rh, rw = max(1, int(H * self.hp_ratio // 2)), max(1, int(W * self.hp_ratio // 2))
        
        mask = torch.ones((H, W), device=student_feat.device)
        mask[cy-rh:cy+rh, cx-rw:cx+rw] = 0
        
        s_high = s_fft_shift * mask
        t_high = t_fft_shift * mask
        
        # 4. Loss on Magnitude
        # We use Log-scale or squared magnitude to emphasize differences
        loss = F.mse_loss(torch.abs(s_high), torch.abs(t_high))
        
        return loss
