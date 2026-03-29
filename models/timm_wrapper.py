import torch
import timm

class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0},
                 checkpoint_path: str = None,
                 pool: bool = True):
        super().__init__()
        self.model = timm.create_model(model_name, **kwargs)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
