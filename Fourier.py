import torch
import torch.nn as nn
from einops import rearrange

class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8, max_N=10):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_N = max_N

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes):
        B, N, _ = boxes.shape
        xyxy_embedding = self.fourier_embedder(boxes)  # B * max_N * C
        objs = self.linears(xyxy_embedding)
        assert objs.shape == torch.Size([B, self.max_N, self.out_dim])
        return objs


class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)

    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)




if __name__ == "__main__":
    net = PositionNet(64, 1024)
    # Adjust max_N as needed
    boxes = torch.randn(32,10, 4)
    out = net(boxes)
    print(out.shape)

