from typing import List
import torch
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, networks: List[nn.Module]):
        super().__init__()
        self.networks = nn.ModuleList(networks)

    def __len__(self):
        return len(self.networks)

    def __getitem__(self, idx):
        return self.networks[idx]

    def forward(self, batch):
        results = [net(batch) for net in self.networks]
        results = {
            k: torch.stack([r[k].squeeze() for r in results], dim=-1)
            for k in results[0].keys()
        }

        return results
