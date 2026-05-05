import torch
import torch.nn.functional as F
from torch import nn
from .ctc_loss import CTCLoss
from .fakd_loss import FAKDLoss

class DistillationCTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, zero_infinity=False, weight_ctc=1.0, weight_fakd=1.0, weight_kd=0.1, temperature=2.0, alpha=0.5, **kwargs):
        super(DistillationCTCLoss, self).__init__()
        self.ctc_loss = CTCLoss(use_focal_loss=use_focal_loss, zero_infinity=zero_infinity)
        self.fakd_loss = FAKDLoss(weight=weight_fakd, alpha=alpha)
        self.weight_ctc = weight_ctc
        self.weight_kd = weight_kd
        self.temperature = temperature
        # Dùng mean để scale loss đều theo batch và sequence length, tránh nổ Loss
        self.kl_loss = nn.KLDivLoss(reduction='mean')

    def forward(self, predicts, batch):
        """
        predicts: dict from DistillationRecognizer
        batch: input from dataloader
        """
        # Nếu đang ở chế độ đánh giá (chỉ trả về tensor dự đoán của student)
        if not isinstance(predicts, dict):
            return self.ctc_loss(predicts, batch)

        student_predicts = predicts['student_predicts']
        teacher_predicts = predicts['teacher_predicts']
        
        # 1. Tính CTC Loss thông thường trên Student
        ctc_out = self.ctc_loss(student_predicts, batch)
        loss_ctc = ctc_out['loss'] * self.weight_ctc
        
        # 2. Tính FAKD Loss trên Features
        fakd_out = self.fakd_loss(predicts, batch)
        loss_fakd = fakd_out['loss_FAKD']
        
        # 3. Tính KL Divergence Loss (Logits KD) để Student học xác suất của Teacher
        student_log_probs = F.log_softmax(student_predicts / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_predicts.detach() / self.temperature, dim=-1)
        loss_kd = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2) * self.weight_kd
        
        loss_total = loss_ctc + loss_fakd + loss_kd
        
        return {
            'loss': loss_total,
            'loss_ctc': loss_ctc,
            'loss_fakd': loss_fakd,
            'loss_kd': loss_kd
        }
