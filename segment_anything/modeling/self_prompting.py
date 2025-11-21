import torch
from torch import nn
from .common import LayerNorm2d


class denseP_Generator(nn.Module):
    def __init__(self, in_channels, mask_in_chans=16, embed_dim = 256):
        super(denseP_Generator, self).__init__()
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(in_channels, mask_in_chans // 4, kernel_size=3, padding=1, stride=1),
            LayerNorm2d(mask_in_chans // 4),
            nn.GELU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=3, padding=1, stride=1),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
    def forward(self, x):
        return self.mask_downscaling(x)


class sparseP_Generator(nn.Module):
    def __init__(self, embed_dim=256):
        super(sparseP_Generator, self).__init__()
        self.embed = nn.LazyLinear(embed_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = self.embed(x)
        x = self.activation(x)
        return x.squeeze(1)


class TAPG(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(TAPG, self).__init__()
        self.denseP = denseP_Generator(in_channels=in_channels, embed_dim=embed_dim)
        self.sparseP = sparseP_Generator(embed_dim=embed_dim)

    def forward(self, input, class_prob):
        if input.ndim == 3: input = input.unsqueeze(0)

        # Dense prompt generation #
        dense_prompt = self.denseP(input * class_prob)
        
        # Sparse prompt generation #
        sparse_prompt = self.sparseP(input)
        sparse_prompt = torch.stack([sparse_prompt, self.not_a_point_embed.weight], dim=1)
        point_lbl = int(class_prob > 0.5)
        sparse_prompt[:, 0] = sparse_prompt[:, 0] + self.point_embeddings[point_lbl].weight

        return dense_prompt, sparse_prompt
