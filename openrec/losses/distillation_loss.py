import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DistillationLoss(nn.Module):
    def __init__(self, 
                 models_loss_config, 
                 distill_loss_config=None,
                 base_loss_name='CTCLoss'):
        """
        Combined loss for Distillation.
        Calculates base losses (e.g. CTC) for each model and optional distillation loss.
        """
        super(DistillationLoss, self).__init__()
        from openrec.losses import build_loss
        
        self.loss_func_dict = nn.ModuleDict()
        for loss_cfg in models_loss_config:
            name = list(loss_cfg.keys())[0]
        self.distill_loss_func = None
        if distill_loss_config is not None:
            self.distill_loss_name = distill_loss_config.get('name', 'DistillKL')
            self.distill_weight = distill_loss_config.get('weight', 1.0)
            self.teacher_name = distill_loss_config.get('teacher_name', 'Teacher')
            self.student_name = distill_loss_config.get('student_name', 'Student')
            self.temperature = distill_loss_config.get('temperature', 1.0)
            self.feat_weight = distill_loss_config.get('feat_weight', 0.0)
            
        self.loss_weights = {}
        for loss_cfg in models_loss_config:
            name = list(loss_cfg.keys())[0]
            self.loss_func_dict[name] = build_loss(loss_cfg[name])
            self.loss_weights[name] = loss_cfg[name].get('weight', 1.0)
            
    def forward(self, predicts, batch):
        total_loss = 0
        loss_dict = {}
        
        # 1. Calculate base losses for each model (e.g. CTC Loss)
        for name, loss_func in self.loss_func_dict.items():
            if name in predicts:
                l = loss_func(predicts[name], batch)
                loss_dict[name] = l['loss']
                # Apply weight to each model's loss
                total_loss += l['loss'] * self.loss_weights[name]
                
        # 2. Calculate Response Distillation Loss (KL Divergence on Logits)
        if hasattr(self, 'distill_loss_name'):
            t_logits = predicts[self.teacher_name]
            s_logits = predicts[self.student_name]
            
            # Apply Temperature scaling
            T = self.temperature
            t_probs = F.softmax(t_logits / T, dim=-1)
            s_log_probs = F.log_softmax(s_logits / T, dim=-1)
            
            # F.kl_div expects log_probs for input and probs for target
            # Scale by T^2 to keep gradients consistent
            distill_loss = F.kl_div(s_log_probs, t_probs, reduction='batchmean') * (T * T)
            
            loss_dict['distill_loss'] = distill_loss * self.distill_weight
            total_loss += loss_dict['distill_loss']
            
        # 3. Calculate Feature Distillation Loss (MSE on Backbone Features)
        if hasattr(self, 'feat_weight') and self.feat_weight > 0:
            t_feat = predicts[f"{self.teacher_name}_feat"]
            s_feat = predicts[f"{self.student_name}_feat"]
            
            feat_loss = F.mse_loss(s_feat, t_feat)
            loss_dict['feat_loss'] = feat_loss * self.feat_weight
            total_loss += loss_dict['feat_loss']
            
        loss_dict['loss'] = total_loss
        return loss_dict
