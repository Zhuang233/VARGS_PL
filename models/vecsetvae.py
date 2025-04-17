# Copyright (c) 2025, Biao Zhang.

import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import OmegaConf

from einops import repeat

import math
import numpy as np
# import mcubes
from torchmcubes import marching_cubes
import trimesh

from utils.utils_vecset import PreNorm, Attention, FeedForward, subsample
from utils.utils_vecset import PointEmbed
from .bottleneck import Bottleneck, KLBottleneck, NormalizedBottleneck

def calc_iou(output, labels, threshold):
    target = torch.zeros_like(labels)
    target[labels>=threshold] = 1
    
    pred = torch.zeros_like(output)
    pred[output>=threshold] = 1

    accuracy = (pred==target).float().sum(dim=1) / target.shape[1]
    accuracy = accuracy.mean()
    intersection = (pred * target).sum(dim=1)
    union = (pred + target).gt(0).sum(dim=1) + 1e-5
    iou = intersection * 1.0 / union
    iou = iou.mean()
    return iou

def points_gradient(inputs, outputs):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad
class TransposedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 用正常的 row-major layout 定义
        self.weight = nn.Parameter(torch.empty(in_features, out_features))  # [256, 1]
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias  # x: [B, N, 256] @ [256, 1] => [B, N, 1]


class VecSetAutoEncoder(LightningModule):
    def __init__(
        self,
        config,
        *,
        depth=24,
        dim=768,
        output_dim=1,
        num_inputs=2048,
        num_latents=1280,
        latent_dim=16,
        dim_head=64,
        query_type='point',
        bottleneck=None,
    ):
        super().__init__()
        self.config = OmegaConf.create(config)
        queries_dim = dim
        
        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents
        
        self.query_type = query_type
        if query_type == 'point':
            pass
        elif query_type == 'learnable':
            self.latents = nn.Embedding(num_latents, dim)
        else:
            raise NotImplementedError(f'Query type {query_type} not implemented')

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = dim // dim_head, dim_head = dim_head)),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)
        
        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = dim // dim_head, dim_head = dim_head)),
                PreNorm(dim, FeedForward(dim))
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = dim // dim_head, dim_head = dim_head))

        # self.to_outputs = nn.Sequential(
        #     nn.LayerNorm(queries_dim),
        #     nn.Linear(queries_dim, output_dim)
        # )
        self.to_outputs = nn.Sequential(
            nn.LayerNorm(queries_dim),
            TransposedLinear(queries_dim, output_dim)
        )
        
        nn.init.zeros_(self.to_outputs[1].weight)
        nn.init.zeros_(self.to_outputs[1].bias)
        
        self.bottleneck = bottleneck

        


    def encode(self, pc):
        B, N, _ = pc.shape
        assert N == self.num_inputs
        
        if self.query_type == 'point':
            sampled_pc = subsample(pc, N, self.num_latents)
            x = self.point_embed(sampled_pc)
        elif self.query_type == 'learnable':
            x = repeat(self.latents.weight, 'n d -> b n d', b = B)
            
        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context = pc_embeddings, mask = None) + x
        x = cross_ff(x) + x

        bottleneck = self.bottleneck.pre(x)
        return bottleneck

    def learn(self, x):

        x = self.bottleneck.post(x)
        
        if self.query_type == 'learnable':
            x = x + self.latents.weight[None]

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        
        return x
    
    def decode(self, x, queries):
        
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        bottleneck = self.encode(pc)
        x = self.learn(bottleneck['x'])

        if queries.shape[1] > 100000:
            N = 100000
            os = []
            for block_idx in range(math.ceil(queries.shape[1] / N)):
                o = self.decode(x, queries[:, block_idx*N:(block_idx+1)*N, :]).squeeze(-1)
                os.append(o)
            o = torch.cat(os, dim=1)
        else:
            o = self.decode(x, queries).squeeze(-1)

        return {'o': o, **bottleneck}

    def training_step(self, batch, batch_idx):
        points, labels, surface, num_vol, num_near, path = batch
        num_vol,num_near = num_vol[0], num_near[0]
        points = points.requires_grad_(True)
        points_all = torch.cat([points, surface], dim=1)
        outputs = self(surface, points_all)
        output = outputs['o']
        grad = points_gradient(points_all, output)

        loss_eikonal = (grad[:, :].norm(2, dim=-1) - 1).pow(2).mean()
        criterion = torch.nn.L1Loss()
        # criterion = torch.nn.BCEWithLogitsLoss()

        loss_vol = criterion(output[:, :num_vol], labels[:, :num_vol])
        loss_near = criterion(output[:, num_vol:num_vol+num_near], labels[:, num_vol:num_vol+num_near])
        loss_surface = (output[:, num_vol+num_near:]).abs().mean()
        loss = loss_vol + 10 * loss_near + 0.001 * loss_eikonal + 1 * loss_surface# + 0.01 * loss_surface_normal
        loss_value = loss.item()
        self.log("train_loss", loss_value, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log("loss_vol", loss_vol, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log("loss_near", loss_near, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log("loss_eikonal", loss_eikonal, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log("loss_surface", loss_surface, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        
        threshold = 0
        vol_iou = calc_iou(output[:, :1024], labels[:, :1024], threshold)
        near_iou = calc_iou(output[:, 1024:2048], labels[:, 1024:2048], threshold)
        return loss

    def validation_step(self, batch, batch_idx):
        # pass
        points, labels, surface, num_vol, num_near, path = batch
        density = 256
        gap = 2. / density
        x = np.linspace(-1, 1, density+1)
        y = np.linspace(-1, 1, density+1)
        z = np.linspace(-1, 1, density+1)
        xv, yv, zv = np.meshgrid(x, y, z)
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(
            np.float32)).view(3, -1).transpose(0, 1)[None].cuda()
        
        outputs = self(surface[0].unsqueeze(0), grid)
        output = outputs['o']

        volume = output.view(density+1, density+1, density+1).permute(1, 0, 2) * (-1)

        verts, faces = marching_cubes(volume.contiguous(), 0)
        verts *= gap
        verts -= 1.
        verts = verts.cpu()
        faces = faces.cpu()
        m = trimesh.Trimesh(verts, faces)

        m.export(f'{path[-2][0]}_{self.current_epoch}.ply')
        # num_vol,num_near = num_vol[0], num_near[0]
        # with torch.enable_grad():
        #     points = points.requires_grad_(True)
        #     points_all = torch.cat([points, surface], dim=1)
        #     outputs = self(surface, points_all)
        #     output = outputs['o'].contiguous()
        #     grad = points_gradient(points_all, output)

        # loss_eikonal = (grad[:, :].norm(2, dim=-1) - 1).pow(2).mean()
        # criterion = torch.nn.L1Loss()
        # # criterion = torch.nn.BCEWithLogitsLoss()
        # loss_near = criterion(output[:, :num_near], labels[:, :num_near])
        # loss_surface = (output[:, num_near:]).abs().mean()
        # loss = 10 * loss_near + 0.001 * loss_eikonal + 1 * loss_surface# + 0.01 * loss_surface_normal
        # self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        points, labels, surface, num_vol, num_near, path= batch
        density = 256
        gap = 2. / density
        x = np.linspace(-1, 1, density+1)
        y = np.linspace(-1, 1, density+1)
        z = np.linspace(-1, 1, density+1)
        xv, yv, zv = np.meshgrid(x, y, z)
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(
            np.float32)).view(3, -1).transpose(0, 1)[None].cuda()
        
        outputs = self(surface[0].unsqueeze(0), grid)
        output = outputs['o']

        volume = output.view(density+1, density+1, density+1).permute(1, 0, 2) * (-1)

        verts, faces = marching_cubes(volume, 0)
        verts *= gap
        verts -= 1.
        verts = verts.cpu()
        faces = faces.cpu()
        m = trimesh.Trimesh(verts, faces)

        m.export(f'output_{path[-2][0]}.ply')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["lr"])
