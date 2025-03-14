import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, input_feature, output_feature, rank=1, alpha=1, device="cpu"):
        super().__init__()
        self.A = nn.Parameter(torch.zeros((input_feature, rank), device=device)) 
        self.B = nn.Parameter(torch.zeros((rank, output_feature), device=device))  
        self.scale = alpha / rank
        self.enabled = True  
        
    def forward(self, wts):
        if self.enabled:
            return wts + torch.matmul(self.A, self.B) * self.scale 
        else:
            return wts 
