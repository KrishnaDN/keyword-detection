from torch import nn
import torch

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(out_dim, in_dim),
                    nn.Dropout(0.1)
                    )
    def forward(self, x):
        return self.block(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.pre_layer_norm = nn.LayerNorm(dim)
        self.post_layer_norm = nn.LayerNorm(dim)
        self.token_mixer = LinearBlock(num_patches, dim)
        self.channel_mixer = LinearBlock(dim, dim)
                            
    def forward(self, x):
        pre_ln =self.pre_layer_norm(x)
        tm_out = self.token_mixer(pre_ln.transpose(1,2)).transpose(1,2)
        tm_out = tm_out + x
        post_ln = self.post_layer_norm(tm_out)
        cm_out = self.channel_mixer(post_ln)+tm_out
        return cm_out