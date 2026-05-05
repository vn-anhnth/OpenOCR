import torch
from torch import nn
from .base_recognizer import BaseRecognizer

class DistillationRecognizer(nn.Module):
    def __init__(self, config):
        """
        Wrapper cho Knowledge Distillation.
        Chứa cả Teacher và Student.
        """
        super(DistillationRecognizer, self).__init__()
        
        # Khởi tạo Student
        student_config = config['Student']
        self.student = BaseRecognizer(student_config)
        
        # Khởi tạo Teacher (Freeze)
        teacher_config = config['Teacher']
        self.teacher = BaseRecognizer(teacher_config)
        
        # Load weights cho Teacher
        if 'pretrained' in teacher_config:
            pretrained_path = teacher_config['pretrained']
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.teacher.load_state_dict(checkpoint['state_dict'])
            else:
                self.teacher.load_state_dict(checkpoint)
        
        # Freeze Teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Projection layer nếu Teacher và Student khác số lượng channels ở đầu ra Encoder
        student_feat_dim = student_config['Encoder']['dims'][-1]
        teacher_feat_dim = teacher_config['Encoder']['dims'][-1]
        
        if student_feat_dim != teacher_feat_dim:
            self.project_layer = nn.Conv2d(student_feat_dim, teacher_feat_dim, kernel_size=1)
        else:
            self.project_layer = nn.Identity()

    def forward(self, x, data=None):
        # 1. Chạy qua Student
        if self.student.use_transform:
            x_s = self.student.transform(x)
        else:
            x_s = x
            
        student_feat = self.student.encoder(x_s)
        student_predicts = self.student.decoder(student_feat, data=data)
        
        # 2. Chạy qua Teacher (chỉ khi đang train)
        if self.training:
            with torch.no_grad():
                if self.teacher.use_transform:
                    x_t = self.teacher.transform(x)
                else:
                    x_t = x
                teacher_feat = self.teacher.encoder(x_t)
                teacher_predicts = self.teacher.decoder(teacher_feat, data=data)
            
            # Project student feature lên cùng size với teacher
            student_feat_proj = self.project_layer(student_feat)
            
            return {
                'student_predicts': student_predicts,
                'student_feat': student_feat_proj,
                'teacher_predicts': teacher_predicts,
                'teacher_feat': teacher_feat
            }
        else:
            # Lúc eval/infer thì chỉ chạy student
            return student_predicts
