import torch
import torch.nn as nn
import torch.fft

class FAKDLoss(nn.Module):
    def __init__(self, weight=1.0, alpha=0.5, **kwargs):
        """
        Frequency-Aware Knowledge Distillation Loss.
        weight: Hệ số tổng của FAKD Loss.
        alpha: Ngưỡng lọc tần số. alpha càng lớn thì chỉ giữ lại tần số càng cao.
        """
        super(FAKDLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def create_high_pass_filter(self, shape, device):
        # shape: (B, C, H, W)
        B, C, H, W = shape
        center_h, center_w = H // 2, W // 2
        
        # Tạo mask lọc (0 ở giữa cho low-frequency, 1 ở ngoài cho high-frequency)
        Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        distance = torch.sqrt((X - center_w)**2 + (Y - center_h)**2)
        
        # Ngưỡng cutoff phụ thuộc vào kích thước feature map
        max_dist = torch.sqrt(torch.tensor(center_h**2 + center_w**2, dtype=torch.float32, device=device))
        cutoff = self.alpha * max_dist
        
        mask = (distance >= cutoff).float()
        return mask.view(1, 1, H, W).expand(B, C, H, W)

    def forward(self, predicts, batch):
        """
        predicts: dict chứa student_feat và teacher_feat từ DistillationRecognizer
        batch: đầu vào từ dataloader (không dùng trong hàm này nhưng cần có để match args)
        """
        student_feat = predicts['student_feat']
        teacher_feat = predicts['teacher_feat']
        
        B, C, H, W = student_feat.shape
        
        # 1. Chuyển sang miền không gian tần số bằng 2D FFT
        # Cast sang float32 để tránh lỗi cuFFT với AMP (Half-precision) không hỗ trợ size ko phải là lũy thừa của 2
        fft_student = torch.fft.fftshift(torch.fft.fft2(student_feat.float(), dim=(-2, -1)))
        fft_teacher = torch.fft.fftshift(torch.fft.fft2(teacher_feat.float(), dim=(-2, -1)))
        
        mag_student = torch.abs(fft_student)
        mag_teacher = torch.abs(fft_teacher)
        
        # 2. Tạo High-pass filter để chỉ lấy tần số cao
        hp_mask = self.create_high_pass_filter(student_feat.shape, student_feat.device)
        
        high_freq_student = mag_student * hp_mask
        high_freq_teacher = mag_teacher * hp_mask
        
        # 3. Ép Student học dải tần số cao của Teacher
        loss_freq = self.mse_loss(high_freq_student, high_freq_teacher)
        
        return {'loss_FAKD': loss_freq * self.weight}
