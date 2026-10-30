# -*- coding: utf-8 -*-
# @Author: Simon Dahan
# @Last Modified time: 2021-12-16 10:41:47
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

'''
This file contains the implementation of the ViT model: https://arxiv.org/abs/2010.11929 adapted to the case of surface patching. 
Input data is a sequence of non-overlapping patches. 
'''

import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange

from vit_pytorch.vit import Transformer



class CrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        dim,
        latent_dim,
        depth,
        heads,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.latent_proj = nn.Linear(latent_dim, dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "query_norm": nn.LayerNorm(dim),
                "context_norm": nn.LayerNorm(dim),
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "ff_norm": nn.LayerNorm(dim),
                "ff": nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, queries, latent_tokens):
        # queries: (B, N, dim)
        # latent_tokens (B, K, latent_dim)
        
        context = self.latent_proj(latent_tokens)

        for layer in self.layers:
            q = layer["query_norm"](queries)
            kv = layer["context_norm"](context)

            attended, _ = layer["cross_attn"](
                query=q,
                key=kv,
                value=kv,
                need_weights=False,
            )

            queries = queries + attended
            queries = queries + layer["ff"](
                layer["ff_norm"](queries)
            )

        return self.final_norm(queries)





class SiT(nn.Module):
    def __init__(self, *,
                        dim, 
                        depth,
                        dec_depth,
                        heads,
                        mlp_dim,
                        latent_dim,
                        pool = 'cls', 
                        num_patches = 20,
                        num_channels =4,
                        num_vertices = 2145,
                        dim_head = 64,
                        dropout = 0.,
                        emb_dropout = 0.
                        ):

        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        patch_dim = num_channels * num_vertices

        # inputs has size = b * c * n * v
        # split img into patches, produce linear embeddings
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c n v  -> b n (v c)'),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, latent_dim)
        )
        self.from_latent = nn.Linear(latent_dim, dim)
        self.mask_token = nn.Parameter(
            torch.randn(1, num_patches, dim) * 0.02
        )

        self.dec_pos = nn.Parameter(
            torch.randn(1, num_patches, dim) * 0.02
        )

        self.decoder = CrossAttentionDecoder(
            dim=dim,
            latent_dim=latent_dim,
            depth=dec_depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.to_pixels = nn.Linear(dim, patch_dim)

        self.rearrange_out = Rearrange('b n (v c) -> b c n v', c=num_channels)
        

    def encode(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.encoder(x)

        # get rid of cls token
        x = x[:, 1:, :]

        x = self.to_latent(x)

        return x

    def decode(self, latent_tokens):
        b = latent_tokens.shape[0]

        queries = self.mask_token.expand(b, -1, -1)
        queries = queries + self.dec_pos

        z = self.decoder(queries, latent_tokens)
        z = self.to_pixels(z)
        return self.rearrange_out(z)

    def forward(self, x):
        return self.decode(self.encode(x))
