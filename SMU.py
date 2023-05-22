import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, List, Optional

class SMU(nn.Module):
    def __init__(self, mu=1000000, alpha=0.5):
        super(SMU, self).__init__()

        self.mu = torch.tensor(mu, dtype=torch.float32)

        self.alpha = torch.tensor(alpha, dtype=torch.float32)



    def forward(self, x):
        return ((1 + self.alpha) * x + (1 - self.alpha) * x * torch.erf(self.mu * (1 - self.alpha) * x)) / 2