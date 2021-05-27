import torch.nn as nn
import torch
from .base import BaseModel
from .keyword_transformer import Transformer, Attention
from argparse import Namespace
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .mlp_mixer import MixerBlock

class KWSTransformer(BaseModel):
    def __init__(self,params):
        super(KWSTransformer, self).__init__()
        var = Namespace(**params['model_params'])
        num_patches = int(var.input_size[0]/var.patch_size[0] * var.input_size[1]/var.patch_size[1])
        patch_dim = var.channels * var.patch_size[0] * var.patch_size[1]
        assert var.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = var.patch_size[0], p2 = var.patch_size[1]),
            nn.Linear(patch_dim, var.dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, var.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, var.dim))
        self.dropout = nn.Dropout(var.emb_dropout)
        self.transformer = Transformer(var.dim, var.depth, var.heads, var.dim_head, var.mlp_dim, var.dropout)
        self.pool = var.pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(var.dim),
            nn.Linear(var.dim, var.num_classes)
        )
        self.crit = nn.CrossEntropyLoss()
        
    
    def forward(self, inputs, target):
        x = self.to_patch_embedding(inputs)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        probs = self.mlp_head(x)
        loss = self.compute_loss(probs, target)
        return loss, torch.topk(probs,1)[1].squeeze(1)
        
    def compute_loss(self, probs, target):
        loss = self.crit(probs, target)
        return loss

    def inference(self, inputs):
        x = self.to_patch_embedding(inputs)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        probs = self.mlp_head(x)
        return torch.topk(probs,1)[1].squeeze(1)


class MLPMixer(nn.Module):
    def __init__(self, params):
        super(MLPMixer, self).__init__()
        var = Namespace(**params['model_params'])
        
        assert (var.input_size[0] % var.patch_size[0]) == 0, 'H must be divisible by patch size'
        assert (var.input_size[1] % var.patch_size[1]) == 0, 'W must be divisible by patch size'
        num_patches = int(var.input_size[0]/var.patch_size[0] * var.input_size[1]/var.patch_size[1])
        patch_dim = var.channels * var.patch_size[0] * var.patch_size[1]
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = var.patch_size[0], p2 = var.patch_size[1]),
            nn.Linear(patch_dim, var.dim),
         )   
        self.network = nn.Sequential(*[nn.Sequential(MixerBlock(var.dim,num_patches)) for _ in range(var.depth)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(var.dim, var.num_classes)
        self.crit = nn.CrossEntropyLoss()
        
        
    def forward(self,inputs, targets):
        x = self.to_patch_embedding(inputs)
        x = self.network(x)
        probs = self.classifier(self.pool(x.transpose(1,2)).squeeze(2))
        loss = self.compute_loss(probs, targets)
        return loss, torch.topk(probs,1)[1].squeeze(1)

    
    def compute_loss(self, probs, target):
        loss = self.crit(probs, target)
        return loss

    def inference(self, inputs):
        x = self.to_patch_embedding(inputs)
        x = self.network(x)
        probs = self.classifier(self.pool(x.transpose(1,2)).squeeze(2))
        return torch.topk(probs,1)[1].squeeze(1)





class MatchBoxNet(BaseModel):
    def __init__(self,params):
        super(MatchBoxNet, self).__init__()
        print('MatchBoxNet')
        pass


class MHAAtnnRNN(BaseModel):
    def __init__(self,params):
        super(MHAAtnnRNN, self).__init__()
        print('MHAAtnnRNN')
        pass


class TCResNet(BaseModel):
    def __init__(self,params):
        super(TCResNet, self).__init__()
        print('TCResNet')
        pass


class Res15(BaseModel):
    def __init__(self,params):
        super(Res15, self).__init__()
        print('Res15')
        pass
    
    
class DSCNN(BaseModel):
    def __init__(self,params):
        super(DSCNN, self).__init__()
        print('DSCNN')
        pass
    
    

