import torch
from torch import nn
from torch.nn import functional as F
from gptx.configs.llama_config import Config


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj_o = nn.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias
        )
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc_1 = self.fc_1(x)
        x_fc_2 = self.fc_2(x)
        x = F.silu(x_fc_1) * x_fc_2
        return self.proj_o(x)
