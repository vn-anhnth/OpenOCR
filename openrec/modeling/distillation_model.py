import torch
import torch.nn as nn
import copy

class DistillationModel(nn.Module):
    def __init__(self, config):
        """
        Distillation Model wrapper that holds multiple models (Teacher, Student).
        """
        super(DistillationModel, self).__init__()
        from openrec.modeling import build_model
        
        self.model_list = nn.ModuleDict()
        for model_cfg in config['models']:
            name = list(model_cfg.keys())[0]
            params = model_cfg[name]
            
            # Extract pretrained path if exists
            pretrained = params.pop('pretrained', None)
            
            model = build_model(params)
            
            if pretrained is not None:
                checkpoint = torch.load(pretrained, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded {name} from {pretrained}")
            
            # Freeze teacher if specified
            if params.pop('freeze', False):
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()
                print(f"Frozen {name}")
                
            self.model_list[name] = model
            
    def forward(self, x, data=None):
        if self.training:
            result = {}
            for name, model in self.model_list.items():
                # Get encoder features and final predictions
                # BaseRecognizer forward: x = self.encoder(x), then x = self.decoder(x)
                feat = model.encoder(x)
                preds = model.decoder(feat, data=data)
                
                result[name] = preds
                result[f"{name}_feat"] = feat
            return result
        else:
            # For evaluation, we only need the Student's prediction
            return self.model_list['Student'](x, data=data)
